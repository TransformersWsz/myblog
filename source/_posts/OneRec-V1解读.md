---
title: OneRec-V1解读
mathjax: true
toc: true
date: 2026-10-30 17:15:30
updated: 2026-10-30 17:15:30
categories:
- 搜广推
tags:
- LLM
---

OneRec通过端到端的生成式框架替代了传统的召回-排序的级联架构。在此系统学习下OneRec-V1。

![overview](https://github.com/TransformersWsz/picx-images-hosting/raw/master/image.c40617.png)

<!--more-->

## 1. 背景与动机

传统推荐系统采用**多阶段级联架构**（召回 → 粗排 → 精排 → 重排），论文指出其三大根本性缺陷：

### 1.1 计算碎片化
以快手为例，服务期 **超过 50% 的资源用于通信和存储**而非高精度计算。精排模型在旗舰 GPU 上的 MFU 仅 **4.6%（训练）/ 11.2%（推理）**，而 LLM 在 H100 上可达约 40%。同时 QPS > 400k、延迟 < 500ms 的约束，使模型被迫在低算力规模运行。

### 1.2 目标冲突
- **多目标冲突**：CTR、观看时长之外还有数百个来自用户、创作者、平台生态的目标，分散在各阶段优化，逐步破坏系统一致性；
- **跨阶段建模冲突**：召回效果受限于排序模型能力，排序又受上游次优结果影响。

### 1.3 落后于 AI 演进
LLM/VLM 领域的缩放定律、强化学习（RLHF/DPO/GRPO）等突破，因级联架构的结构性壁垒难以迁移到推荐场景。

### 1.4 OneRec 的解法
> 将召回与排序**合并为单阶段 Encoder-Decoder 生成式框架**，端到端直接优化最终目标，并严格优化训练/推理计算效率。



## 2. 整体架构

OneRec 是一个**Encoder-Decoder生成模型**：输入用户多尺度行为表示，自回归生成目标视频的语义 ID 序列。

![encoder-decoder](https://github.com/TransformersWsz/picx-images-hosting/raw/master/image.133a1b.png)

整体流程如下：

```
用户特征/行为序列
      │
      ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐     ┌──────────────┐
│  Tokenizer  │     │   Encoder    │     │     Decoder     │     │ Reward System│
│ 视频→语义ID │     │ 用户兴趣压缩 │     │ MoE 逐Token生成 │     │ P-Score/格式 │
└─────────────┘     └──────────────┘     └─────────────────┘     └──────────────┘
                                                      │
                                                      ▼
                                        语义ID → 视频映射 → (可选)RM Selection
```

## 3. 核心组件详解

### 3.1 Tokenizer：协同感知多模态语义 ID

**方案**：融合多模态特征以及协同过滤信号，用 **RQ-Kmeans** 生成由粗到细的层次化语义 ID。

#### 3.1.1 多模态表示

**多模态表示**：每个视频输入 caption、tag、ASR、OCR、封面图、5 帧均匀采样帧，用 miniCPM-V-8B 处理，得到 $N_M = 1280$ 个 token 向量 $\mathbf{M} \in \mathbb{R}^{N_M \times d_t}$（$d_t = 512$）。再用 QFormer 以 $N_{\tilde{M}} = 4$ 个可学习 query token 压缩：

$$\mathbf{Q}^{(i+1)} = \mathrm{CrossAttn}\left(\mathbf{Q}^{(i)},\mathbf{M},\mathbf{M}\right)$$

$$\mathbf{Q}^{(i+1)} = \mathrm{FFN}(\mathrm{RMSNorm}(\mathbf{Q}^{(i+1)})),\quad \text{for } i\in\{1,2,\ldots,N_c\}$$

其中压缩表示 $\tilde{\mathbf{M}} = \mathbf{Q}^{(N_c+1)} \in \mathbb{R}^{N_{\tilde{M}} \times d_t}$，$N_c = 4$ 为 QFormer 层数。

#### 3.1.2 协同过滤先验

**Item 对构造** $\mathcal{D}_{pair}$：
1. User-to-Item 检索：对每个用户，取正点击目标 item，与用户最近历史正点击中最协同相似的 item 配对；
2. Item-to-Item 检索：使用Swing之类的I2I且大于一定分数的Item组成Pairs

**双训练目标**：

1. Item-to-Item 对比学习：

$$
\mathcal{L}_{I2I} = -\frac{1}{|\mathcal{B}|}\sum_{(i,j)\in \mathcal{B}}\log \frac{\exp \left(\mathrm{sim}\left(\tilde{\mathbf{M}}_i,\tilde{\mathbf{M}}_j\right) / \tau\right)}{\sum_{(i',j')\in \mathcal{B}}\exp \left(\mathrm{sim}\left(\tilde{\mathbf{M}}_i,\tilde{\mathbf{M}}_j\right) / \tau\right)}
$$

2. Caption 生成损失（防止幻觉，保留内容理解能力，用 LLaMA3 作解码器）：

$$\mathcal{L}_{caption\_gen} = -\sum_k\log P\left(t^{k+1}\mid\left[t^1,t^2,\dots,t^k\right]\right)$$

其中 $\tau$ 为温度系数，$\mathrm{sim}(\cdot,\cdot)$ 为相似度函数，$\mathcal{B}$ 为 $\mathcal{D}_{pair}$ 的一个 batch，$t^k$ 为第 $k$ 个 caption token。

#### 3.1.3 RQ-Kmeans 量化

初始残差（$l=1$）：

$$
\mathcal{R}^{(1)} = \{\tilde{\mathbf{M}}_i\in \mathbb{R}^{N_{\tilde{M}}\times d_t}\mid \forall \text{ video } i\}
$$

每层码本由残差的 K-means 质心得到：

$$
C^{(l)} = \mathrm{K\text{-}means}\left(\mathcal{R}^{(l)},N_t\right)
$$

其中 $C^{(l)} = \{\mathbf{c}_k^{(l)}\in \mathbb{R}^{N_{\tilde{M}}\times d_t}\mid k=1,\ldots,N_t\}$，$N_t$ 为码本大小。

最近质心索引：

$$
s_i^l = \arg \min_k\left\| \mathcal{R}_i^{(l)} - \mathbf{c}_k^{(l)}\right\|
$$

残差更新：

$$
\mathcal{R}_i^{(l+1)} = \mathcal{R}_i^{(l)} - \mathbf{c}_{s_i^l}^{(l)}
$$

迭代 $L_t = 3$ 层。最终视频 $m$ 表示为粗到细的语义标识 $\{s_m^1, s_m^2, \ldots, s_m^l\}$，支持渐进式 item 生成。



### 3.2 Encoder：多尺度特征工程 + 用户兴趣压缩

#### 3.2.1 四条行为通路

| 通路 | 序列长度 | 内容 |
|---|---|---|
| 用户静态 | 1 | uid、age、gender 等 |
| 短期行为 | $L_s = 20$ | vid/sid、aid、tag、ts、playtime、dur、label |
| 正反馈行为 | $L_p = 256$ | 同上，高互动序列 |
| 终身行为 | 原始 10 万 → 压缩 $L_l = 2000$ | 层次 K-means + QFormer 压缩 |

**用户静态通路**（$\mathbf{e}\in\mathbb{R}^{64}$）：

$$\mathbf{f}_u = [\mathbf{e}_{\mathrm{uid}};\mathbf{e}_{\mathrm{gender}};\mathbf{e}_{\mathrm{age}};\dots ],\quad \mathbf{h}_u = \mathrm{Dense}(\mathrm{LeakyReLU}(\mathrm{Dense}(\mathbf{f}_u)))$$

其中 $\mathbf{h}_u\in \mathbb{R}^{1\times d_{\mathrm{model}}}$。

**短期通路**：

$$\mathbf{f}_s = [\mathbf{e}_{\mathrm{vid}}^s;\mathbf{e}_{\mathrm{aid}}^s;\mathbf{e}_{\mathrm{tag}}^s;\mathbf{e}_{\mathrm{tu}}^s;\mathbf{e}_{\mathrm{playtime}}^s;\mathbf{e}_{\mathrm{dur}}^s;\mathbf{e}_{\mathrm{label}}^s]$$

$$\mathbf{h}_s = \mathrm{Dense}(\mathrm{LeakyReLU}(\mathrm{Dense}(\mathbf{f}_s)))$$

维度约定：video embedding 对齐 $d_{\mathrm{model}}$，author embedding 512 维，其余 128 维；$\mathbf{h}_s\in \mathbb{R}^{L_s\times d_{\mathrm{model}}}$。

**正反馈通路**（结构同上，序列长度 $L_p = 256$）：

$$\mathbf{f}_p = [\mathbf{e}_{\mathrm{vid}}^p;\mathbf{e}_{\mathrm{aid}}^p;\mathbf{e}_{\mathrm{tag}}^p;\mathbf{e}_{\mathrm{tu}}^p;\mathbf{e}_{\mathrm{playtime}}^p;\mathbf{e}_{\mathrm{dur}}^p;\mathbf{e}_{\mathrm{label}}^p]$$

$$\mathbf{h}_p = \mathrm{Dense}(\mathrm{LeakyReLU}(\mathrm{Dense}(\mathbf{f}_p))),\quad \mathbf{h}_p\in \mathbb{R}^{L_p\times d_{\mathrm{model}}}$$

**终身通路**（最长 10 万条，两阶段层次压缩）：

*行为压缩*：对每个用户交互序列做层次 K-means，每步簇数设为 $\lfloor \sqrt[3]{|D|} \rfloor$（$|D|$ 为当前数据 item 数，经验设定），当簇内 item 数不超过阈值 $M$ 时终止，选离簇中心最近的 item 作代表。

*特征聚合*：稀疏类别特征（vid、aid、label）直接继承代表视频特征；连续特征（tag、ts、playtime、dur）取簇内均值。

长序列（$L_l = 2000$）上每个视频被替换为对应簇代表特征：

$$\mathbf{f}_l = [\mathbf{e}_{\mathrm{vid}}^l;\mathbf{e}_{\mathrm{aid}}^l;\mathbf{e}_{\mathrm{tag}}^l;\mathbf{e}_{\mathrm{ts}}^l;\mathbf{e}_{\mathrm{playtime}}^l;\mathbf{e}_{\mathrm{dur}}^l;\mathbf{e}_{\mathrm{label}}^l]$$

$$\mathbf{v}_l = \mathrm{Dense}(\mathrm{LeakyReLU}(\mathrm{Dense}(\mathbf{f}_l))),\quad \mathbf{v}_l\in \mathbb{R}^{L_l\times d_{\mathrm{model}}}$$

QFormer 压缩（$N_q = 128$ 个可学习 query，$N_l = 2$ 个 block）：

$$\mathbf{h}_l^{(i+1)} = \mathrm{CrossAttn}(\mathbf{h}_l^{(i)},\mathbf{v}_l,\mathbf{v}_l)$$

$$\mathbf{h}_l^{(i+1)} = \mathrm{FFN}(\mathrm{RMSNorm}(\mathbf{h}_l^{(i+1)}))$$

得到压缩终身表示 $\mathbf{h}_l = \mathbf{h}_l^{(N_l)}\in \mathbb{R}^{N_q\times d_{\mathrm{model}}}$。

#### 3.2.2 Encoder 架构

拼接四通路输出并加位置编码：

$$\mathbf{z}^{(1)} = [\mathbf{h}_u;\mathbf{h}_s;\mathbf{h}_p;\mathbf{h}_l] + \mathbf{e}_{\mathrm{pos}}$$

其中 $\mathbf{e}_{\mathrm{pos}}\in \mathbb{R}^{(1+L_s+L_p+N_q)\times d_{\mathrm{model}}}$ 为可学习位置编码。

经过 $L_{\mathrm{enc}}$ 层 Transformer（全可见自注意力 + FFN + RMSNorm）：

$$\mathbf{z}^{(i+1)} = \mathbf{z}^{(i)} + \mathrm{SelfAttn}(\mathrm{RMSNorm}(\mathbf{z}^{(i)}))$$

$$\mathbf{z}^{(i+1)} = \mathbf{z}^{(i+1)} + \mathrm{FFN}(\mathrm{RMSNorm}(\mathbf{z}^{(i+1)}))$$

最终输出 $\mathbf{z}_{\mathrm{enc}} = \mathbf{z}^{(L_{\mathrm{enc}}+1)}\in \mathbb{R}^{(1+L_s+L_p+N_q)\times d_{\mathrm{model}}}$。


### 3.3 Decoder：MoE + 逐点生成

**点式生成范式**：对每个目标视频 $m$，构造 `[BOS, s¹, s², s³]` 输入序列：

$$\mathbf{S}_m = \left\{s_{[\mathrm{BOS}]},s_m^1,s_m^2,\dots ,s_m^L\right\},\quad \mathbf{d}_m^{(0)} = \mathrm{Emb\_lookup}(\mathbf{S}_m)$$

每个 Decoder 层（$L_{\mathrm{dec}}$ 层）：

$$\mathbf{d}_m^{(i+1)} = \mathbf{d}_m^{(i)} + \mathrm{CausalSelfAttn}(\mathbf{d}_m^{(i)})$$

$$\mathbf{d}_m^{(i+1)} = \mathbf{d}_m^{(i+1)} + \mathrm{CrossAttn}(\mathbf{d}_m^{(i+1)},\mathbf{Z}_{\mathrm{enc}},\mathbf{Z}_{\mathrm{enc}})$$

$$\mathbf{d}_m^{(i+1)} = \mathbf{d}_m^{(i+1)} + \mathrm{MoE}(\mathrm{RMSNorm}(\mathbf{d}_m^{(i+1)}))$$

**MoE 层**（$N_{\mathrm{experts}}$ 个专家，top-$k$ 路由）：

$$\mathrm{MoE}(\mathbf{x}) = \sum_{j=1}^{k}\mathrm{Gate}_j(\mathbf{x})\cdot \mathrm{Expert}_j(\mathbf{x})$$

采用 DeepSeek 的无辅助损失负载均衡策略（loss-free balancing），避免干扰梯度。SwiGLU 专家隐维按 $\frac{2}{3}\times 4\times d_{\mathrm{model}}$ 计算并取 128 的倍数。

**NTP 训练损失**：

$$\mathcal{L}_{\mathrm{NTP}} = -\sum_{j=0}^{L_t-1}\log P\left(s_m^{j+1}\mid \left[s_{\mathrm{BOS}},s_m^1,s_m^2,\dots ,s_m^j\right]\right)$$

**模型规格表**：

| 模型 | Layers | Hid. Dim | FFN Hid. Dim | Heads | Experts (Tot/Act) | MoE 位置 |
|---|---|---|---|---|---|---|
| OneRec-0.015B (Dense) | 4 | 128 | 256 | 4 | N/A | N/A |
| OneRec-0.121B (Dense) | 8 | 1024 | 2048 | 8 | N/A | N/A |
| OneRec-0.935B (MoE) | 8 | 1024 | 2048 | 8 | 24 / 2 | Decoder |
| OneRec-2.633B (MoE) | 24 | 1024 | 2048 | 8 | 24 / 4 | Enc & Dec |

> 注：Layers = #Encoder + #Decoder，两者层数相同。


### 3.4 奖励系统与 ECPO 强化学习

预训练模型只拟合曝光 item 分布（来自传统系统），无法突破传统推荐天花板。因此引入基于奖励系统的偏好对齐，在生成 item 空间做 on-policy RL。

#### 3.4.1 偏好对齐：P-Score

用神经网络学习个性化融合分 P-Score（基于 SIM 架构，多塔各学单目标，隐层 + user/item 表示进 MLP）：

$$\mathcal{L}_{\mathrm{P\text{-}Score}} = \sum_{x\mathrm{tr}\in S_o}w^{\mathrm{xtr}}\mathcal{L}_{\mathrm{P\text{-}Score}}^{\mathrm{xtr}}$$

$$\mathcal{L}_{\mathrm{P\text{-}Score}}^{\mathrm{xtr}} = -(y^{\mathrm{xtr}}\log p + (1-y^{\mathrm{xtr}})\log (1-p))$$

$$S_o = \{\mathrm{ctr},\mathrm{lvtr},\mathrm{ltv},\mathrm{vtr},\ldots \}$$

调整 $w^{\mathrm{xtr}}$ 使 P-Score 偏向各目标，最终实现**全目标 AUC 提升**（相比人工加权求和更易达 Pareto 优化）。

#### 3.4.2 ECPO（Early Clipped GRPO）

对用户 $u$，旧策略生成 $G$ 个 item，每个 item 与 user 输入偏好奖励模型得 $r_i$：

$$\mathcal{J}_{ECPO}(\theta) = \mathbb{E}_{u\sim P(U),\{o_i\}_{i=1}^G\sim \pi_{\theta_{old}}}\left[\frac{1}{G}\sum_{i=1}^G\min\left(\frac{\pi_{\theta}(o_i|u)}{\pi_{\theta_{old}}'(o_i|u)}A_i,\mathrm{clip}\left(\frac{\pi_{\theta}(o_i|u)}{\pi_{\theta_{old}}'(o_i|u)},1-\epsilon,1+\epsilon\right)A_i\right)\right]$$

优势估计（组内归一化）：

$$A_i = \frac{r_i - \mathrm{mean}(\{r_1,r_2,\dots,r_G\})}{\mathrm{std}(\{r_1,r_2,\dots,r_G\})}$$

**改进的旧策略**（提前裁剪）：

$$\pi_{\theta_{old}}'(o_i|u) = \max\left(\frac{\mathrm{sg}(\pi_{\theta}(o_i|u))}{1+\epsilon+\delta},\pi_{\theta_{old}}(o_i|u)\right),\quad \delta > 0$$

其中 $\mathrm{sg}$ 为 stop gradient，OneRec 设 $\delta = 0.1$。

**关键改进动机**：原 GRPO 对负优势允许大策略比，易梯度爆炸。ECPO 对负优势样本**提前 clip 最大 ratio**，保证训练稳定同时保留负优势效果；因 RL 与 SFT 联合训练，**移除 KL 散度损失**（SFT loss 保证稳定性）。

#### 3.4.3 格式奖励（Generation Format Regularization）

**挤压效应（Squeezing Effect）**：加入 RL 后，负优势样本使模型把概率质量压缩到当前最优输出 $o^*$，导致合法 token 概率被挤到与非法 token 相当，合法率跌破 50%。

**解法**：从 $G$ 个样本随机选 $K$ 个做合法性 RL：

$$A_i = \begin{cases}1 & \mathrm{if}\ o_i\in I_{\mathrm{legal}}\\ 0 & \mathrm{if}\ o_i\notin I_{\mathrm{legal}}\end{cases}$$

优化目标同 ECPO 公式，直接用 $A_i$ 作优势。实验显示 Top-k 选择会让模型只在 top 子集学合法（整体合法率先升后降），**随机选择**更稳健，最终达 95% 合法率，且带来 App 停留 +0.13%、观看时长 +0.30%。

#### 3.4.4 工业场景对齐：SIR

以打击"营销号/二创"为例，设最优占比 $f$，超出部分对 P-Score 降权：

$$r_i^{\prime} = \begin{cases}r_i & \mathrm{if}\ o_i\notin I_{\mathrm{viral}}\\ \alpha r_i & \mathrm{if}\ o_i\in I_{\mathrm{viral}}\end{cases},\quad \alpha\in(0,1)$$

效果：营销号曝光 **↓9.59%**，核心指标（观看时长、App 停留）稳定。这是传统级联"某阶段打补丁"做不到的端到端一致优化。


## 4. 训练框架

### 4.1 训练基础设施
- **算力**：90 台服务器 × 8 旗舰 GPU + 2 CPU，400Gbps NVLink 节点内互联；
- **网络**：节点内 NVLink，节点间 400Gbps RDMA（训练流量）+ 100Gbps TCP（数据/embedding 预取）；
- **存储**：每台 4 块 NVMe SSD 加速 checkpoint 写入，embedding 与 dense 参数存 HDFS；
- **加速优化**：SKAI GPU 参数服务器（跨 GPU 统一 embedding 表 + GPU 缓存 + 预取）、数据并行 + ZeRO1 + 梯度累积、BF16 混合精度、attention 编译优化。

### 4.2 预训练
- 每样本目标 item 量化为 **3 个语义 ID**，即每样本 3 个目标 token；
- 日处理约 **180 亿样本**，吞吐 **540 亿 token/天**；
- OneRec-0.935B 在约 **1000 亿样本**（3000 亿 token 曝光）后收敛。

### 4.3 后训练
- 实时数据流在线训练，**RSFT + RL 同时进行**；
- RSFT：按播放时长过滤掉底部 50% 曝光 session，loss 同 $\mathcal{L}_{\mathrm{NTP}}$，稀疏参数 lr 退火至 $1\times10^{-4}$、稠密至 $8\times10^{-5}$；
- RL：随机选 1% 用户生成 RL 样本；
- **解耦生成与训练**：1% 用户访问外部推理服务生成 512 个 item 并请求奖励，每 1000 步通过 MQ 把更新参数推给外部服务。


## 5. 评估结果

### 5.1 评估指标
- **NTP loss**：下一 token 预测损失曲线；
- **P-Score**：学习的综合偏好分；
- **xtr 指标**：由线上预训练排序模型给出的 lvtr（长播率）、vtr（播放率）、ltr（点赞率）、wtr（关注率）、cmtr（评论率）等预测概率。

### 5.2 缩放定律

**参数缩放**：0.015B → 0.121B → 0.935B → 2.633B，loss 持续下降；数据量在约 100 亿样本内快速收敛，之后增益放缓但不完全平台化（超 1000 亿样本仍有收益）。MoE 负载均衡策略带来 **0.2 的 loss 下降**。

**特征缩放**（0.935B 模型）：

| 指标 | w/o feature | w/ feature | 提升 |
|---|---|---|---|
| lvtr | 0.4940 | 0.5500 | 11.34% |
| vtr | 0.8730 | 0.8901 | 1.96% |
| ltr | 0.0391 | 0.0441 | 12.79% |
| wtr | 0.0190 | 0.0224 | 17.89% |
| cmtr | 0.0919 | 0.1010 | 9.90% |
| P-score | 0.0749 | 0.0966 | 28.88% |

**码本缩放**（8192 → 32768）：

| 指标 | Size=8K | Size=32K | 提升 |
|---|---|---|---|
| lvtr | 0.5118 | 0.5245 | 2.48% |
| vtr | 0.9384 | 0.9491 | 1.14% |
| ltr | 0.0298 | 0.0299 | 0.34% |
| wtr | 0.0153 | 0.0154 | 0.65% |
| cmtr | 0.0650 | 0.0664 | 2.15% |
| P-score | 0.2516 | 0.2635 | 4.75% |

**推理缩放**（Pass@K）：

| 指标 | Pass@8 | Pass@64 | Pass@512 | Pass@1024 | 提升 |
|---|---|---|---|---|---|
| lvtr | 0.3675 | 0.4927 | 0.5351 | 0.5443 | 48.11% |
| vtr | 0.9444 | 0.9462 | 0.9513 | 0.9530 | 0.91% |
| ltr | 0.0278 | 0.0346 | 0.0425 | 0.0452 | 62.59% |
| wtr | 0.0114 | 0.0138 | 0.0182 | 0.0197 | 72.81% |
| cmtr | 0.0350 | 0.0566 | 0.0809 | 0.0891 | 154.57% |
| P-score | 0.0811 | 0.2051 | 0.3375 | 0.3859 | 376.10% |

> K 从 512 → 1024 增益边际，生产环境选 **K=512**。

### 5.3 语义 ID 作为输入表示

2.633B 规模下，用语义 ID 替代 vid 稀疏 embedding，性能持平或略优：

| 指标 | VID | Semantic ID | 提升 |
|---|---|---|---|
| lvtr | 0.4447 | 0.4467 | 0.45% |
| vtr | 0.8725 | 0.8726 | 0.01% |
| ltr | 0.0336 | 0.0336 | 0.00% |
| wtr | 0.0104 | 0.0105 | 0.96% |
| cmtr | 0.0565 | 0.0573 | 1.42% |
| P-score | 0.0371 | 0.0378 | 1.74% |

**四大优势**：参数效率（省十亿级 vid 稀疏表）、通信效率（稠密操作替代参数服务器通信）、序列容量（可扩展到万级交互）、表示一致性（输入输出共享语义空间）。

### 5.4 强化学习消融

**采样效率**（不同 Pass@K 下 RL 影响）：

| 方法 | vtr | Watch time | App Stay Time | Video View |
|---|---|---|---|---|
| Pass@32 w/o RL | 0.1978 | +1.62% | -0.10% | -4.18% |
| Pass@32 w/ RL | 0.2138 | +3.17% | +0.39% | -9.87% |
| **相对提升** | **+8.08%** | **+1.55%** | **+0.49%** | -3.69% |
| Pass@128 相对提升 | +6.61% | +1.53% | +0.38% | -2.65% |
| Pass@512 相对提升 | +2.05% | -0.41% | +0.09% | +1.97% |

> RL 在 Pass@32 提升最大，说明 top 排序精度显著增强，利于降低采样开销。

**搜索空间**（group size，Pass@128 下）：

| Group Size | vtr | Watch time | App Stay Time | Video View |
|---|---|---|---|---|
| 0 (w/o RL) | 0.2198 | +4.61% | +1.11% | -12.75% |
| 128 | 0.2303 | +5.22% | +1.49% | -15.06% |
| 512 | 0.2350 | +5.73% | +1.82% | -15.49% |
| 2048 | 0.2352 | +5.84% | +1.78% | -15.49% |

> 128 → 512 显著提升，2048 无额外收益。建议 ECPO group size ≈ 推理输出量的 **4 倍**。

**搜索策略**：

| 策略 | vtr | Watch time | App Stay Time | Video View |
|---|---|---|---|---|
| Top-k + Top-p | 0.2131 | +4.45% | +1.16% | -13.61% |
| Beam Search | 0.2162 | +5.35% | +1.76% | -13.30% |
| 相对提升 | +1.45% | +0.87% | +0.60% | +0.36% |

> 语义 ID 的前缀树编码结构与 Beam Search 系统性探索高度契合。

**参考模型**：on-policy（当前策略模型）离线奖励更优（vtr 0.2389 vs 0.2262），但线上提升不显著（存在轻微 reward hacking），是未来重点方向。

**P-Score 奖励效果**：

| 场景 | Watch time | App Stay Time | Video View |
|---|---|---|---|
| 快手 | +0.21% | +0.26% | +0.17% |
| 快手极速版 | +0.71% | +0.22% | +0.35% |

### 5.5 线上 A/B 测试

5% 流量实验组，观察一周，主指标 App 停留时长和 LT7。两组：纯生成模型（OneRec）与生成 + 奖励模型选择（OneRec with RM Selection）。

| 场景 | 指标 | OneRec | OneRec + RM Selection |
|---|---|---|---|
| **快手** | App Stay Time | +0.01% | **+0.54%** |
| | Watch Time | +0.07% | +1.98% |
| | Video View | +1.98% | +2.52% |
| | Like | -2.00% | +2.43% |
| | Follow | -2.88% | +3.24% |
| | Comment | -1.56% | +5.27% |
| | Collect | -0.61% | +2.93% |
| | Forward | +0.27% | +5.90% |
| **快手极速版** | App Stay Time | +0.06% | **+1.24%** |
| | Watch Time | +0.05% | +3.28% |
| | Video View | +2.40% | +3.39% |
| | Like | -2.64% | +1.49% |
| | Follow | -2.75% | +2.28% |
| | Comment | -2.23% | +3.20% |
| | Collect | -1.76% | +1.91% |
| | Forward | -1.86% | +3.48% |

> 纯生成模型已"匹配整个复杂推荐系统"的性能，但交互指标为负；加 RM Selection 后全面转正，无跷跷板效应。已扩至约 25% 总 QPS。

**vs 禁用缓存对照组**（更严格基线）：快手 App 停留 +0.20%、LT7 +0.03%；极速版 +0.55%、+0.02%，且 LT7 增长趋势显著更强。

**本地生活场景**：GMV +21.01%、订单量 +17.89%、买家数 +18.58%、新买家获取 +23.02%，已接管该业务 100% QPS。

### 5.6 推理基础设施
NVIDIA L20 GPU，每台 4 GPU + 2 CPU（PCIe），UniPredict 平台，200Gb RDMA（RoCE，跨机最高 800Gb），TensorRT 编译 + 自定义 cross-attention/MoE 插件 + batching + MPS，吞吐提升 5×，MFU 达 28.8%。


## 6. 结论、局限与未来方向

**结论**：OneRec 以 Encoder-Decoder 生成式架构压缩用户终身行为、用 MoE 扩展解码器容量、用定制 RL 框架对齐奖励函数。MFU 达 23.7%/28.8%，OPEX 仅传统系统 10.6%，效果与效率全面超越现有系统。

**局限**：
1. **推理阶段缩放**：step scaling 尚不明显，缺乏强推理能力；
2. **多模态集成**：尚未与 LLM/VLM 深度结合，用户行为模态有望成为原生多模态输入；
3. **奖励系统设计**：仍很初步，但端到端架构为奖励系统的技术突破创造了条件。

**未来**：OneRec 已作为快手基础范式，系统性推进算法创新与团队协作机制，构建支撑大规模流量增长的可扩展基础设施。

---

## 参考

- [细读快手推荐OneRec-V1技术报告](https://zhuanlan.zhihu.com/p/1927386141486735715)