---
title: Uplift Tree建模
mathjax: true
toc: true
date: 2024-11-25 01:39:49
updated: 2024-11-25 01:39:49
categories:
- Marketing
tags:
- Uplift
- Tree
---

决策树构建流程：

<!--more-->

### 1. 节点划分
在每个树节点上，遍历所有特征以及该特征下的所有取值，以最大化实验组和对照组之间的转化率差异。

### 2. 增益计算
每个节点根据实验组和对照组在该特征下的转化率差异来计算增益。例如，某一群体可能在干预下表现出更高的购买率，而另一群体则没有显著变化。增益计算公式如下：
$$
D_{\text {gain }}=D_{\text {after }\_{\text {split }}}\left(P^T(Y), P^C(Y)\right)-D_{\text {before}\_{\text {split }}}\left(P^T(Y), P^C(Y)\right)
$$

- $P^T(Y)$表示实验组样本中类别$Y$的概率
- $P^C(Y)$表示对照组样本中类别$Y$的概率
- $D$表示度量距离，可以欧氏距离、KL散度等
- $D_{\text {after }\_{\text {split }}}\left(P^T(Y), P^C(Y)\right)$ 即为该叶子节点的uplift值，一般都是$P^T(Y) - P^C(T)$

### 3. 增益预测
给定一用户，根据该用户的所有特征以及该特征下的所有取值，将其分配到某一个叶子节点，得到该用户的uplift值。

___

## 参考
- [因果推断笔记——uplift建模、meta元学习、Class Transformation Method（八）](https://cloud.tencent.com/developer/article/1913905)
- [闲聊因果效应（5）：树模型（Tree-Based）、分类模型（The class transformation）](https://zhuanlan.zhihu.com/p/636342238)