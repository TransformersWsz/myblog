---
title: 'SGM: Sequence Generation Model for Multi-Label Classification'
mathjax: true
toc: true
date: 2025-03-09 02:44:40
updated: 2025-03-09 02:44:40
categories:
- NLP
tags:
- Multi-label Classification
- LSTM
---
为了建模多标签之间的依赖关系，本篇工作用序列生成的方式来解决该问题。

<!--more-->

当前label的预测不仅依赖于输入上下文，也依赖于已输出的所有label。用seq2seq建模标签依赖是一种非常自然的思路，但存在如下两大问题：

1. 序列建模强调标签的先后顺序，即位置关系，而多标签是一个集合，不存在位置约束，哪个标签在前在后没有关系，只要输出正确就行。这种情况下，ground truth该如何构造？
2. 序列生成是自回归形式，当前label的生成依赖于上一个label，如果上一个label是错误的，那么将会严重影响后续所有label的预测。这种情况下，该减轻预测错误的label所导致的连锁反应？

SGM针对上述问题提出了如下建模思路：

## 模型结构

经典的序列生成范式：
$$
p(\boldsymbol{y} \mid \boldsymbol{x})=\prod_{i=1}^n p\left(y_i \mid y_1, y_2, \cdots, y_{i-1}, \boldsymbol{x}\right)
$$

![model](https://github.com/TransformersWsz/picx-images-hosting/raw/master/image.1e8r4ljlun.webp)

#### 问题1的解法
作者根据训练集中标签出现的频次来构造标签序列：高频标签置前，低频标签置后。同时在序列头尾插入 `bos` 和 `eos` 表示序列的开始与结束。

#### 问题2的解法
引入Global Embedding考虑所有可能label的信息，避免贪心依赖上一个label：
$$
\overline{\boldsymbol{e}}=\sum_{i=1}^L y_{t-1}^{(i)} \boldsymbol{e}_i \\
g\left(\boldsymbol{y}_{t-1}\right)=(\mathbf{1}-\boldsymbol{H}) \odot \boldsymbol{e}+\boldsymbol{H} \odot \overline{\boldsymbol{e}} \\
\boldsymbol{H}=\boldsymbol{W}_1 \boldsymbol{e}+\boldsymbol{W}_2 \overline{\boldsymbol{e}}
$$

$y_{t-1}$是在$t-1$时间步预测的标签概率分布，$e_i$是$l_i$的embedding。本质上就是根据概率分布对所有可能标签做加权求和。$H$则是门控机制，控制加权embedding的比例。

## 实验结果

![exp](https://github.com/TransformersWsz/picx-images-hosting/raw/master/image.2yyi43ptta.webp)

加上GE效果更加明显！

___

## 参考
- [SGM](https://aclanthology.org/C18-1330.pdf)
- [多标签文本分类-如何有效的利用标签之间的关系](https://zhuanlan.zhihu.com/p/377543518?utm_psn=1879874767093494719)
- [多标签分类新建模方法](https://transformerswsz.github.io/2024/03/18/%E5%A4%9A%E6%A0%87%E7%AD%BE%E5%88%86%E7%B1%BB%E6%96%B0%E5%BB%BA%E6%A8%A1%E6%96%B9%E6%B3%95/)