---
title: OneRec-V2解读
mathjax: true
toc: true
date: 2026-11-01 21:41:10
updated: 2026-11-01 21:41:10
categories:
- 搜广推
tags:
- LLM
---

OneRec-V1 虽然在线上取得了一定的收益，但面临两大瓶颈：

1. **计算效率低下**：Encoder-Decoder 架构中 97.66% 的计算资源消耗在 context encoding（用户上下文编码）上，而非真正参与 loss 计算的 target decoding（目标视频生成）上。
2. **RL 依赖代理奖励模型**：V1 的 P-Score 奖励模型训练依赖老系统样本，存在 reward hacking 风险和采样效率问题。

针对这两个问题，V2 提出了两大核心改进：**Lazy Decoder-Only 架构** + **基于真实用户反馈的偏好对齐**。

<!--more-->

![overview](https://github.com/TransformersWsz/picx-images-hosting/raw/master/image.9cda08.png)

## 一、Lazy Decoder-Only 架构

### 1.1 问题：Encoder-Decoder 的计算碎片化

V1 使用标准的 Encoder-Decoder 架构：Encoder 负责把用户上下文编码成上下文表示，Decoder 用 cross-attention 读取上下文并自回归生成语义 ID。

但论文做了详细的 FLOPs 拆解（Table 1），当 context length N=3000 时：

| 架构 | 总计算 (GFLOPs) | Context Encoding (GFLOPs) | Target Decoding (GFLOPs) | Target 占比 |
|------|-----------------|--------------------------|-------------------------|------------|
| Encoder-Decoder (0.5B:0.5B) | 1988 | 1980 | 8.1 | 0.41% |
| Naive Decoder-Only (1B) | 3618 | 3600 | 18 | 0.49% |
| **Lazy Decoder-Only (1B)** | **18** | **~0** | **18** | **≈100%** |

可以看到，无论是 Encoder-Decoder 还是 Naive Decoder-Only，绝大部分计算都花在了 context encoding 上（>99%），而真正参与 loss 计算的 target decoding 仅占不到 1%。

> ⚠️注意：V2分配给target的算力没有变化，只是占比提升了。

### 1.2 核心设计：Context Processor + Lazy Cross-Attention

V2 的核心思想是：**上下文只需要编码一次，然后所有 decoder layer 共享这份编码**。

![lazy decoder](https://github.com/TransformersWsz/picx-images-hosting/raw/master/image.718ef1.png)

**Context Processor**：将用户多路特征（静态路径、短期路径、长期路径）通过线性层投影后，做一次 RMSNorm，然后生成 layer-specific 的 key-value pairs。

具体地，context tensor 沿特征维度切分为 $L_{kv}$ 组：

$$
\text{Context} = [C_0, C_1, \ldots, C_{S_{kv} \cdot L_{kv} - 1}]
$$

对于每一层 $l \in \{0, 1, \ldots, L_{kv} - 1\}$，计算归一化的 key-value pairs：

$$
k_l = \text{RMSNorm}_{k,l}(C_{l \cdot S_{kv}})
$$

$$
v_l = \begin{cases}
\text{RMSNorm}_{v,l}(C_{l \cdot S_{kv} + 1}), & \text{if } S_{kv} = 2 \text{ (separated key-value)} \\
k_l, & \text{if } S_{kv} = 1 \text{ (shared representation)}
\end{cases}
$$

实验发现 $S_{kv} = 1$（即 key 和 value 共享同一份表示）时 loss 不变但激活值减少。

**Lazy Cross-Attention**：在 cross-attention 中不再做 K/V 的投影计算（因为 Context Processor 已经做过了），直接复用已计算的 KV pairs。同时引入 **GQA（Grouped Query Attention）**——query heads 保持完整数量，但 key-value heads 共享，进一步减少 KV 内存和访存带宽。

### 1.3 效果对比

| 架构 | 总参数 | GFLOPs | Activations | 收敛 Loss |
|------|--------|--------|-------------|----------|
| Enc:Dec=1:1 (V1) | 1B | 296.36 | 17.63B | 3.28 |
| Naive Dec-Only | 1B | 634.83 | 31.53B | * |
| **Lazy Dec-Only (V2)** | **1B** | **18.89** | **1.24B** | **3.27** |

V2 相比 V1：**FLOPs 降低 15.7 倍，激活值降低 14.2 倍**，收敛 loss 还略优（3.27 vs 3.28）。

更直观地看计算比例的变化：

| 架构 | Context Encoding 占比 | Target Decoding 占比 |
|------|----------------------|---------------------|
| Encoder-Decoder (V1) | 99.6% | 0.41% |
| Naive Decoder-Only | 99.5% | 0.49% |
| **Lazy Decoder-Only (V2)** | **≈0%** | **≈100%** |

V2 把计算资源从"无意义的重复上下文编码"中彻底解放出来，全部投入到真正参与 loss 计算的 target decoding 上。

### 1.4 GQA 的进一步优化

在 Lazy Decoder 基础上，GQA 进一步减少了 cross-attention 中的 KV 大小和激活内存：

| Gkv | GFLOPs | Activations | KV Size | 收敛 Loss |
|-----|--------|-------------|---------|----------|
| 14 | 18.89 | 1.24B | 94M | 3.27 |
| 7 | 18.74 | 1.19B | 47M | 3.28 |
| 2 | 18.64 | 1.16B | 13M | 3.28 |
| **1** | **18.62** | **1.15B** | **7M** | **3.27** |

Gkv=1 时 KV 大小从 94M 压缩到 7M（降低 13.4 倍），而 loss 不变。


## 二、训练数据组织：避免数据泄露

V2 重新审视了训练数据的组织方式，指出了 V1 可能存在的问题：

- **Naive Impression Organization（V1 可能用的）**：把所有 impression 的序列拼在一起训练。问题在于，同一个用户在不同时间的 impression 之间可能存在模式重复（如 A→B 在多次 impression 中重复出现），导致过拟合。
- **User-Centric Organization（V2 采用）**：按用户为中心组织训练数据，确保训练时不会用到"未来"的信息。例如在 t3 时刻训练 User-2 的数据时，不能用到 t4 时刻 User-1 的交互（因为那对 User-2 来说是"未来"）。
- **New Impression Only**：只训练最新一次的 impression，进一步避免数据泄露和模式重复。

这个改动虽然不显眼，但对训练稳定性和泛化能力很重要。


## 三、RL 奖励信号：从 Proxy 到 Real

### 3.1 Duration-Aware Reward Shaping（时长感知奖励塑形）

V1 的 P-Score 是一个多目标融合的神经网络奖励模型，训练标签来自老系统曝光后的点击/点赞/观看时长等反馈。V2 没有放弃奖励模型，而是**额外引入了基于真实用户交互信号的直接奖励**，并做了关键改进：

**问题**：原始播放时长（play time）有严重的时长偏差——长视频天然更容易获得高播放时长，但这不代表用户更喜欢。

**V2 的解法**：

1. 对用户历史观看视频按**对数时长分桶**（因为视频时长服从长尾分布）
2. 对于目标视频，计算其播放时长在**同桶历史视频中的分位数**，作为用户的偏好分数 $q_i$

公式表达：

$$
q_i = \frac{|\{p_j \in P_{bucket} : p_j \leq p_i\}|}{|P_{bucket}|}
$$

其中 $P_{bucket}$ 是该用户历史视频中与目标视频时长在同一 log 桶内的播放时长集合。

**效果**：这个分位数分数消除了视频时长的系统性偏差，更准确地反映用户真实偏好。

### 3.2 正负样本构造

V2 还引入了负信号：

$$
A_i = \begin{cases}
+1, & q_i > \tau_B \text{ and } n_{neg,i} = 0 \\
-1, & n_{neg,i} = 1 \\
0, & \text{otherwise}
\end{cases}
$$

即：
- 高偏好分数（超过阈值 $\tau_B$）且无负反馈 → 正样本 ($A_i = +1$)
- 有负反馈 → 负样本 ($A_i = -1$)
- 其他 → 中性 ($A_i = 0$)

这比 V1 只用正样本更丰富，能更好地学习用户偏好边界。

### 3.3 User Feedback Signals vs Reward Model

V2 做了一个很有意义的消融实验（Table 7），对比三种奖励信号策略：

| 策略 | 说明 |
|------|------|
| **Reward Model** | 用代理奖励模型（类似 V1 的 P-Score） |
| **User Feedback Signals** | 用真实用户交互信号（Duration-Aware Reward） |
| **Hybrid** | 两者混合 |

**快手主 App 结果：**

| 指标 | Reward Model | User Feedback | Hybrid |
|------|-------------|---------------|--------|
| App Stay Time | +0.269% | +0.299% | +0.283% |
| Watch Time | +0.537% | +0.610% | +0.118% |
| Video View | +0.505% | +0.647% | +0.647% |
| Like | +6.552% | +2.435% | +7.010% |
| Follow | +7.265% | +2.007% | +8.458% |
| Comment | +15.472% | +0.944% | +8.763% |
| Collect | +1.856% | +1.401% | +9.739% |
| Forward | +12.024% | +0.803% | +5.270% |

**关键发现：**

- **User Feedback Signals** 在 Watch Time 上最强（+0.610%），说明时长感知奖励确实有效
- **Reward Model** 在互动指标上更强（Like +6.55%、Follow +7.27%、Comment +15.47%），因为 P-Score 本身就是多目标融合的，能平衡各指标
- **Hybrid** 虽然单项不是最高，但整体最均衡——App Stay Time 和互动指标都有不错提升，没有明显的跷跷板效应

论文的解释：Reward Model 的奖励是多目标融合的（ctr、lvtr、ltr 等），而 User Feedback 主要基于播放时长，所以两者偏好方向不同。混合使用可以兼顾时长和互动。

---

## 四、GBPO：梯度有界策略优化

### 4.1 V1 ECPO 的问题

V1 用的是 ECPO（Early Clipped GRPO），对负 advantage 的样本做 early clipping（限制策略比率上限），防止梯度爆炸。但论文发现：

- 对于传统流水线产生的样本（非 OneRec 生成），由于无法获取真实 $\pi_{old}$，只能简化为 $\pi_{old} = \text{sg}(\pi_\theta)$，导致策略比率恒为 1
- 传统方法认为比率为 1 的样本训练稳定、不做截断
- 但实际上，**负样本即使比率为 1 也可能导致梯度爆炸**（论文 Figure 9 的梯度范数曲线证实了这一点）

### 4.2 GBPO 的核心思想

GBPO 从 BCE 损失的梯度分析出发：

$$
\frac{\partial \mathcal{L}_{BCE}}{\partial \theta} = \begin{cases}
-\frac{1}{p_\theta}\frac{\partial p_\theta}{\partial \theta}, & y=1 \\
\frac{1}{1-p_\theta}\frac{\partial p_\theta}{\partial \theta}, & y=0
\end{cases}
$$

关键洞察：**对于负样本（y=0），当前模型概率 $p_\theta$ 越小，抑制时的梯度越小，训练更稳定**。

基于此，GBPO 对策略比率的约束改为**动态边界**：

$$
\pi'_{\theta_{old}}(o_i|u) = \begin{cases}
\max(\pi_{\theta_{old}}, \text{sg}(\pi_\theta)), & A_i \geq 0 \\
\max(\pi_{\theta_{old}}, 1 - \text{sg}(\pi_\theta)), & A_i < 0
\end{cases}
$$

对比传统 clip 方法（固定边界 $[1-\epsilon, 1+\epsilon]$），GBPO 的三个区别：

1. **不丢弃任何样本的梯度**（传统 clip 会在边界外产生"no gradient"区域）
2. **负样本的边界是动态的**，依赖于当前 $\pi_\theta$ 而非固定常数
3. **正样本保持与 GRPO/ECPO 一致**，不做额外约束

### 4.3 为什么叫"Gradient-Bounded"？

因为 GBPO 本质上是用 BCE 损失的梯度上界来约束 RL 梯度——BCE 梯度的稳定性（负样本概率越小时梯度越小）被借用过来约束 RL 策略更新的幅度，从而保证训练稳定性而不需要粗暴地 clip。

---

## 五、最终线上效果：V2 vs V1

Table 8 展示了 OneRec-V2 相比 OneRec-V1 的线上 A/B 测试结果：

![result](https://github.com/TransformersWsz/picx-images-hosting/raw/master/image.021b56.png)


所有指标全面超越 V1，且没有明显的跷跷板效应。

### 关闭缓存的极端场景验证

论文 Appendix D 还做了一个关闭缓存的实验（Table 9），所有流量都请求 OneRec-V2 实时推理：

- 互动指标提升更显著（Like +15%~+30%、Comment +17%~+29%）
- **但冷启视频观看量大幅下降**（快手 -44.7%、极速版 -36.7%）
- 聚类密度上升（+11.7% / +7.9%），说明推荐更集中到热门内容

这个结果很有意思：**去掉缓存后，模型更倾向于推荐"安全的"热门内容**，冷启视频曝光被严重挤压。这其实呼应了之前讨论中提到的问题——RL 训练依赖老系统样本时，模型会偏向"老系统验证过的内容分布"。V2 虽然引入了真实用户反馈信号，但冷启问题仍未完全解决。

## 总结
V2 的核心思路是"架构瘦身 + 奖励去代理化"——用 Lazy Decoder-Only 把计算效率拉到极致（全部花在 target decoding 上），用真实用户反馈信号替代纯代理奖励模型来引导 RL，同时用 GBPO 保证训练稳定性。最终在快手主 App 和极速版上相比 V1 获得了 App Stay Time +0.47%~0.74% 的额外提升，所有互动指标全面进步。但冷启视频曝光下降的问题仍然值得关注。

___

## 参考
- [细读快手OneRec V2技术报告](https://zhuanlan.zhihu.com/p/1945765458063098684)