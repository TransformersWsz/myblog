---
title: Flash-Attention
mathjax: true
toc: true
date: 2024-02-19 02:20:52
categories:
- Machine Learning
tags:
- LLM
- Transformer
- Attention
---

这是一篇硬核的优化Transformer的工作。众所周知，Transformer模型的计算量和储存复杂度是 $O(N^2)$ 。尽管先前有了大量的优化工作，比如LongFormer、Sparse Transformer、Reformer等等，一定程度上减轻了Transformer的资源消耗，但对Transformer的性能有所折损，且扩展性不强，不能泛化到其它领域、以及复杂结构的叠加。

<!--more-->

这篇工作从底层对Transformer的计算和读写进行了优化，主要有三个贡献：

1. 加速了模型计算：现在GPU的计算速度已经远远超过了内存读写速度(***模型计算速度慢是因为IO慢，而不是$O(N^2)$的原因导致。也就是说transformer的瓶颈在IO，而不是运算***)，当GPU完成计算后，内存却还在读取数据，造成GPU闲置而内存繁忙读（消费者早就消费完了，生产者还在缓慢生产）的现象，也就是内存墙问题。FlashAttention通过tiling和算子融合计算，将复杂操作放到SRAM中计算，并减少从HBM读取次数，加快了模型计算速度。而之前的工作虽然减少了Transformer的计算复杂度，却并没有减少模型计算时间。
2. 节省了显存：FlashAttention通过引入全局统计量，避免实例化大注意力矩阵，减少了显存占用。
3. 精确的注意力：FlashAttention从底层优化了Transformer的计算，但是任务指标上没有任何折损，与普通的Transformer结果是完全等价。

## 现代GPU内存分级

![GPU](https://github.com/TransformersWsz/picx-images-hosting/raw/master/image.3j7dpa8fd1g0.webp)

flash attention的思路就是尽量地在SRAM中进行分块计算、算子融合，减少对HBM（即常说的显存）的读写，从加快模型计算，减轻内存墙问题。

## 算法流程
![algorithm](https://raw.githubusercontent.com/TransformersWsz/picx-images-hosting/master/image.4s3gxobimia0.webp)

## tiling分块计算
```python
# ---------------------
# Tc: K和V的分块数
# Tr: Q的分块数量
# ---------------------
for 1 <= j <= Tc:
    for 1 <= i <= Tr:
        do....
```

![loop](https://raw.githubusercontent.com/TransformersWsz/picx-images-hosting/master/image.3msrg3wcmqq0.webp)

由于对$Q, K$矩阵进行了分块，就无法进行全局归一化。我们的最终目的是得到 $O$ ，作者这里根据公式推导，不断用当前最新的rowmax和rowsum去更新，直到遍历完最后一块，最终结果就和标准场景下的结果完全一致。

## 计算量和显存分析

1. 计算量：$O(N^2 d)$，跟标准attention计算一致

![computation](https://raw.githubusercontent.com/TransformersWsz/picx-images-hosting/master/image.6zirlknz7m80.webp)

2. 显存：$m \in R^N, l \in R^N$
   
![gpu memory](https://raw.githubusercontent.com/TransformersWsz/picx-images-hosting/master/image.2ysn4gh16f80.webp)

## IO复杂度分析

![IO](https://raw.githubusercontent.com/TransformersWsz/picx-images-hosting/master/image.68jimy8oo9s0.webp)

- 标准attention需要加载 $Q \in R^{N \times d}, K \in R^{N \times d}$ 以及 $S \in R^{N \times N}$，所以是 $O(Nd + N^2)$
___

## 参考
- [FlashAttention:加速计算,节省显存, IO感知的精确注意力](https://zhuanlan.zhihu.com/p/639228219)
