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

![mixing & reverting](https://github.com/TransformersWsz/picx-images-hosting/raw/master/image.ed4c8c.png)

1. **Mixing**：将输入 $T \times D$ 的 Token 切分并跨 Token 拼接，打散重组成 $H \times (T \cdot D/H)$，实现全局信息交互。
2. **Reverting**：将混合后的表示，按原始 Token 的位置重新拼接，**还原回 $T \times D$ 的形状**。

此时，再与**原始输入 $\mathbf{X}$** 进行残差相加。
* **为什么有效？** 保证了输入输出维度严格一致，残差加的是语义对齐的原始 $\mathbf{X}$，而不是混合后的错位表示。这使得残差路径完全畅通，原始 Token 信息被完整保留。



#### 改进二：深层网络的“前向 + 反向”双管齐下

**解决痛点**：深层梯度更新不足。

![interval residual](https://github.com/TransformersWsz/picx-images-hosting/raw/master/image.833dc5.png)

**原理解析**：
除了标准的残差连接，引入两个机制：
1. **Interval Residual（间隔残差）**：每隔 2 或 3 层，将低层特征直接跨层加到高层。解决**前向**信息流衰减问题。但注意，最后一层不加，以免干扰高层抽象。
2. **Auxiliary Loss（辅助损失）**：在中间层（间隔 2~3 层）提取输出，过轻量 MLP 得到辅助 logits，与主 logits 联合计算 BCE Loss。
$$
\mathcal{L}_{\text{joint}}
= \underbrace{\mathcal{L}\!\left(\hat{y}^{(L)},\, y\right)}_{\text{Main Loss}}
\;+\;
\sum_{i}\underbrace{\mathcal{L}\!\left(\hat{y}^{(\ell_i)},\, y\right)}_{\text{Aux Loss}_i}
$$

> ⚠️这块论文只有一段话，没有详细展开来讲，语焉不详。目前推测为低维、高维logits均参与loss计算，保证梯度可以传播到低维空间。

#### 改进三：Sparse-Pertoken MoE（稀疏专家混合）

**解决痛点**：Dense 训练太贵，ReLU-MoE 动态激活难以预测。

**原理解析**：
* **先放大，再稀疏（First Enlarge, Then Sparse）**：先设计一个性能极佳的稠密 SwiGLU 模型，然后将大 FC 层物理切分为多个细粒度子专家。训练和推理都只激活 Top-k 个。实现了真正的 **Sparse Train, Sparse Infer**。
* **Shared Expert（共享专家）**：每个 Token 永远激活一个共享专家兜底，提升稳定性。
* **Gate Value Scaling（门控缩放）**：在路由权重前乘以常数 $\alpha$。因为专家被切碎后，每个专家分到的梯度变小了；实验发现 $\alpha$ 与稀疏比成反比（如 1:2 稀疏度 $\alpha=2$），这强行让梯度幅度与稠密模型一致。
* **Down-Matrix Small Init**：将 SwiGLU 中下投影矩阵 $W_{\text{down}}$ 的初始化方差降到 0.01，使其在训练初期接近恒等映射，防止深层输出爆炸。

#### 改进四：纯模型设计（Pure Model Design）

**解决痛点**：碎片化算子拖累。

**原理解析**：
随着模型参数变大（如超过 500M），TokenMixer-Large 自身强大的表达能力，完全能够吸收掉前端诸如 DCN、LHUC 等老旧碎片的收益。因此，**直接移除这些低效算子**。例如，DCN 在 150M 模型上能带来 +0.09% 的 AUC 提升，但在 700M 参数时，移除 DCN 没有任何效果损失。


#### 实验结果
![tokenmixer-result](https://github.com/TransformersWsz/picx-images-hosting/raw/master/image.84d9c7.png)

#### 消融实验
![tokenmixer-ablation](https://github.com/TransformersWsz/picx-images-hosting/raw/master/image.af8217.png)

可以看到`残差维度不匹配与语义错位`导致性能下降最明显，说明其有效保留了输入信号，避免了初始信息的丢失。

### 序列特征交互Scaling
