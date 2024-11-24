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

1. 节点划分：在每个树节点上，遍历所有特征以及该特征下的所有取值，以最大化实验组和对照组之间的行为差异。

增益计算：每个节点根据对照组和实验组在该特征下的行为差异来计算增益。例如，某一群体可能在干预下表现出更高的购买率，而另一群体则没有显著变化。

预测增益：通过在树的每一层分配不同的用户群体，Uplift Tree预测每个用户在接受干预后的行为增益，即他们比没有干预时更有可能采取目标行为（如购买）。

## 参考
- [因果推断笔记——uplift建模、meta元学习、Class Transformation Method（八）](https://cloud.tencent.com/developer/article/1913905)
- [闲聊因果效应（5）：树模型（Tree-Based）、分类模型（The class transformation）](https://zhuanlan.zhihu.com/p/636342238)