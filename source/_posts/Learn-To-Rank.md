---
title: Learn To Rank
mathjax: true
toc: true
date: 2024-07-07 02:50:12
updated: 2024-07-07 02:50:12
categories:
- 搜广推
tags:
- Algorithm
- Neural Networks
---

在信息检索中，给定一个query，搜索引擎召回一系列相关的Documents，然后对这些Documents进行排序，最后将Top N的Documents输出。

<!--more-->

{% note danger %}
排序问题最关注的是各Documents之间的相对顺序关系，而不是各个Documents的预测分最准确。
{% endnote %}

## 训练数据

|方法|人工标注|行为日志|
|:---:|:---:|:---:|
|简介|人工对抽样出来作为training data的query-doc pair进行相关程度的判断和标注|根据用户的实际搜索和点击行为，来判断query-doc的相关性。比如同一个query下，不同doc的点击数来作为它们相关程度的大小|
|优点|准确性高|无须人工干预，成本低|
|缺点|代价高且耗时|用户行为日志存在大量偏差，比如：<li>位置偏差：用户倾向于点击列表靠前的item</li><li>样本选择偏差：有用户点击的query知识总体query的一个子集，无法获取全部的query下doc的label</li>|

## 评价指标
这里主要介绍[NDCG](https://chatgpt.com/share/613f6af0-fdc1-4435-81e0-8c3a3b763779)

## 三大rank算法
### pointwise
pointwise方法损失函数计算只与单个document有关，本质上是训练一个分类模型或者回归模型，判断这个document与当前的这个query相关程度，最后的排序结果就是从模型对这些document的预测分值进行一个排序。

- 优点：实现简单
- 缺点：
  - 精确打分，而不是相对打分，无法实现排序
  - 损失函数也没有建模到预测排序中的位置信息

### pairwise
pairwise方法在计算目标损失函数的时候，每一次需要基于一个pair的document的预测结果进行损失函数的计算。其中模型输入和对应的标签label形式如下：
- 输入：一个文档对(docA, docB)
- 输出：相对序(1 or 0.5 or 0)

- 优点：实现简单；建模了两个文档相对序关系
- 缺点
  - 样本对量级高，$O(n^2)$
  - 对错误标注数据敏感，会造成多个pair对错误
  - 仅考虑了文档对pair的相对位置，仍然没有建模到预测排序中的位置信息

#### 经典模型RankNet
![RankNet](https://raw.githubusercontent.com/TransformersWsz/picx-images-hosting/master/image.8ojkqhz95u.png)

### listwise
Listwise方法是直接对整个list的document的排序进行优化，目标损失函数中优化整个list的document的排序结果。其中模型输入和对应的标签label形式如下：
- 输入: 整个list document
- 输出: 排序好的document list

- 优点：直接建模list内的所有文档序关系，与评估目标一致
- 缺点
  - 计算复杂度高

#### 经典模型ListMLE
直接以真实标签顺序为目标，最大化预测结果排序与目标一致的概率即可。
![ListMLE](https://raw.githubusercontent.com/TransformersWsz/picx-images-hosting/master/image.7egnk999fg.png)

___

## 参考
- [Learning to Rank简介](https://www.cnblogs.com/bentuwuying/p/6681943.html)
- [learning to rank中的Listwise，Pairwise和Pointwise](https://xdren69.github.io/2021/04/26/learning-to-rank/)
- [Learning to Rank : ListNet与ListMLE](https://blog.csdn.net/qq_36478718/article/details/122598406)