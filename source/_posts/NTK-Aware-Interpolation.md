---
title: NTK-Aware Interpolation
mathjax: true
toc: true
date: 2024-04-30 02:14:41
updated: 2024-04-30 02:14:41
categories:
- NLP
tags:
- LLM
- 长度外推
- RoPE
---

主要思路：高频外推，低频内插。

$$
m \theta_i=m *(\text { base } * \alpha)^{-2 i / d}=m *(10000 * \alpha)^{-2 i / d}
$$

<!--more-->

NTK的优点是不用微调的情况下，能比线性插值做得好。但是由于低频部分还是会有部分被外推到超出范围的值，因此在设定系数的时候，要比需要的设得更大才行。

## 参考

- [大模型处理长上下文方法一览](https://mp.weixin.qq.com/s/81NHGf5W8HEscW2dBK8MRg)
- [详解基于调整RoPE旋转角度的大模型长度外推方法](https://zhuanlan.zhihu.com/p/670280576)