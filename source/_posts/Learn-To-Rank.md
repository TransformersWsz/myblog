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
- 人工标注：比较准确，但代价高且耗时
- 搜索日志：根据用户的实际搜索和点击行为，来判断query-doc的相关性。比如同一个query下，不同doc的点击数来作为它们相关程度的大小
  - 


https://xdren69.github.io/2021/04/26/learning-to-rank/



___

## 参考
- [Learning to Rank简介](https://www.cnblogs.com/bentuwuying/p/6681943.html)