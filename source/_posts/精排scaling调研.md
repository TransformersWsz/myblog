---
title: 精排scaling调研
mathjax: true
toc: true
date: 2026-11-05 01:43:41
updated: 2026-11-05 01:43:41
categories:
- 搜广推
tags:
- Scaling
- Transformer
---

集中调研下Scaling在搜广推领域的进展。

<!--more-->

# 0. 背景：推荐精排为什么缺 Scaling Law

LLM 的公式是：**统一的 Transformer block + 堆参数堆数据 = 效果可预测地涨**。推荐精排模型做不到，卡点在输入异构：

## 非序列特征交互Scaling

### TokenMixer-Large

RankMixer虽然表现出色，但在试图进一步Scale up时，遇到了以下几个关键瓶颈：

1. **残差设计次优（语义错位）**：RankMixer 在混合机制中，直接将混合前和混合后的 Token 进行 Add & Norm 相加。由于混合后的 Token 已经将所有原始信息打散重组，两者在 Token 维度上**语义不再对齐**，导致残差连接无法有效传递梯度。
2. **模型架构“不纯”**：历史迭代遗留了大量碎片化算子（如 LHUC、DCNv2）。这些算子计算强度低但访存高，严重拖累整体效率。
3. **深层梯度更新不足**：原版通常只堆叠 2 层左右，一旦加深，残差路径受限，梯度消失问题严重，难以稳定训练。
4. **MoE 稀疏化不彻底**：采用“稠密训练，稀疏推理（Dense Train, Sparse Infer）”，训练成本并未降低；且使用的 ReLU-MoE 激活专家数动态变化，对推理极不友好。
5. **规模受限**：受限于上述问题，参数规模仅停留在 1B 左右。

TokenMixer-Large针对上述问题做了系统性的重构。整体模型图如下：

![tokenmixer](https://github.com/TransformersWsz/picx-images-hosting/raw/master/image.99e45f.png)

#### 改进一：Mixing & Reverting（混合与还原）机制

**解决痛点**：残差维度不匹配与语义错位（TSA）。

**原理解析**：
论文提出了对称的“混合-还原”双层结构。
* **第一层（Mixing）**：将输入 $T \times D$ 的 Token 切分并跨 Token 拼接，打散重组成 $H \times (T \cdot D/H)$，实现全局信息交互。
* **第二层（Reverting）**：将混合后的表示，按原始 Token 的位置重新拼接，**还原回 $T \times D$ 的形状**。

此时，再与**原始输入 $\mathbf{X}$** 进行残差相加。
* **为什么有效？** 保证了输入输出维度严格一致，残差加的是语义对齐的原始 $\mathbf{X}$，而不是混合后的错位表示。这使得残差路径完全畅通，原始 Token 信息被完整保留。



### 改进二：深层网络的“前向 + 反向”双管齐下

**解决痛点**：深层梯度更新不足。

**原理解析**：
除了标准的残差连接，引入两个机制：
1. **Interval Residual（间隔残差）**：每隔 2 或 3 层，将低层特征直接跨层加到高层。解决**前向**信息流衰减问题。但注意，最后一层不加，以免干扰高层抽象。
2. **Auxiliary Loss（辅助损失）**：在中间层（间隔 2~3 层）提取输出，过轻量 MLP 得到辅助 logits，与主 logits 联合计算 BCE Loss。
$$
 \mathcal{L}_{\text{total}} = \mathcal{L}_{\text{main}} + \lambda \sum_{l \in \mathcal{S}} \text{BCE}(\hat{y}_l, y) 
$$
   这相当于强迫低层自己也能预测目标（估计高层偏差），为低层参数提供**短路径的反向梯度**，防止低层“躺平”。

### 改进三：Sparse-Pertoken MoE（稀疏专家混合）

**解决痛点**：Dense 训练太贵，ReLU-MoE 动态激活难以预测。

**原理解析**：
* **先放大，再稀疏（First Enlarge, Then Sparse）**：先设计一个性能极佳的稠密 SwiGLU 模型，然后将大 FC 层物理切分为多个细粒度子专家。训练和推理都只激活 Top-k 个。实现了真正的 **Sparse Train, Sparse Infer**。
* **Shared Expert（共享专家）**：每个 Token 永远激活一个共享专家兜底，提升稳定性。
* **Gate Value Scaling（门控缩放）**：在路由权重前乘以常数 $\alpha$。因为专家被切碎后，每个专家分到的梯度变小了；实验发现 $\alpha$ 与稀疏比成反比（如 1:2 稀疏度 $\alpha=2$），这强行让梯度幅度与稠密模型一致。
* **Down-Matrix Small Init**：将 SwiGLU 中下投影矩阵 $W_{\text{down}}$ 的初始化方差降到 0.01，使其在训练初期接近恒等映射，防止深层输出爆炸。

### 改进四：纯模型设计（Pure Model Design）

**解决痛点**：碎片化算子拖累。

**原理解析**：
随着模型参数变大（如超过 500M），TokenMixer-Large 自身强大的表达能力，完全能够吸收掉前端诸如 DCN、LHUC 等老旧碎片的收益。因此，**直接移除这些低效算子**。例如，DCN 在 150M 模型上能带来 +0.09% 的 AUC 提升，但在 700M 参数时，移除 DCN 没有任何效果损失。

---

## 三、 实验结果与消融分析

### 1. 全面超越 SOTA（离线指标）

在约 500M 参数规模的电商场景中，TokenMixer-Large 在相同 FLOPs 预算下，表现全面优于 DLRM、DCNv2、AutoInt，以及此前的 SOTA (RankMixer)。

| 模型 | CTCVR ΔAUC | 稠密参数 | FLOPs/Batch |
| :--- | :--- | :--- | :--- |
| DLRM-MLP | 基线 | 499 M | 125.1 T |
| RankMixer | +0.84% | 567 M | 4.6 T |
| **TokenMixer-Large 500M** | **+0.94%** | **501 M** | **4.2 T** |
| **TokenMixer-Large 4B** | **+1.14%** | **4.6 B** | **29.8 T** |
| **TokenMixer-Large 4B (SP-MoE)** | **+1.14%** | **2.3 B (激活)** | **15.1 T** |

> **亮点**：4B 的 SP-MoE 版本，只激活了 2.3B 的参数，就达到了 4B 稠密模型的同等效果（+1.14%），极大地提升了投入产出比。

### 2. 消融实验：哪些模块最重要？

在 4B 参数规模下进行的模块剥离消融：

| 消融设置 | ΔAUC 变化 |
| :--- | :--- |
| w/o Global Token (全局Token) | -0.02% |
| **w/o Mixing & Reverting (混合还原)** | **-0.27%** |
| w/o Residual (标准残差) | -0.15% |
| w/o Interval Residual & AuxLoss (跨层残差与辅助损失) | -0.04% |
| **Per token SwiGLU → SwiGLU** | **-0.21%** |
| Per token SwiGLU → Per token FFN | -0.10% |

**结论**：
* **Mixing & Reverting 是绝对的核心**，去掉它掉点最严重（-0.27%），证明了原版 RankMixer 残差设计的缺陷是真实存在的。
* **Per-token SwiGLU** 同样关键（-0.21%），说明保留特征异质性的同时引入 SwiGLU 是极佳的组合。

### 3. SP-MoE 内部的零成本优化消融

| 消融设置 | ΔAUC 变化 | 参数变化 | FLOPs 变化 |
| :--- | :--- | :--- | :--- |
| w/o Shared Expert | -0.02% | 0.0% | 0.0% |
| w/o Gate Value Scaling | -0.03% | 0.0% | 0.0% |
| w/o Down-Matrix Small Init | -0.03% | 0.0% | 0.0% |
| **Sparse-Pertoken MoE → 普通 Sparse MoE** | **-0.10%** | 0.0% | 0.0% |

**结论**：这三个操作（共享专家、门控缩放、小初始化）**不增加任何参数和 FLOPs**，属于纯粹的“免费午餐”，但联合起来能带来显著的性能提升。此外，Sparse-Pertoken 结构通过自带路由先验，效果远超标准 MoE。

---

## 四、 线上业务收益

模型在字节跳动多个核心场景上线，覆盖了广告、电商、直播，服务数亿用户。在线基线为 RankMixer-1B / 150M / 500M，TokenMixer-Large 成功扩展至 **7B / 4B / 2B**。

| 业务场景 | 核心指标 | 收益提升 |
| :--- | :--- | :--- |
| 抖音电商 | 订单量 (Order) | **+1.66%** |
| 抖音电商 | 人均预览支付 GMV | **+2.98%** |
| 抖音信息流广告 | 广告主满意度 (ADSS) | **+2.0%** |
| 抖音直播 | 收入 (Revenue) | **+1.4%** |

---

## 总结

TokenMixer-Large 并非推翻 RankMixer，而是通过精准的“外科手术”解决了其规模化道路上的绊脚石。**Mixing & Reverting** 解决了语义对齐，**Interval Residual + AuxLoss** 打通了深层梯度，**先放大再稀疏**的 SP-MoE 提供了极致的 ROI。这套架构证明了：在推荐系统中，抛弃陈旧的碎片化算子，通过结构化的 Token 交互和合理的稀疏激活，我们完全有能力将排序模型推向百亿参数的深水区。

### 序列特征交互Scaling
