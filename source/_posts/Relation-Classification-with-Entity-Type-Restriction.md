---
title: Relation Classification with Entity Type Restriction
mathjax: true
toc: true
date: 2022-05-10 23:48:24
categories:
- NLP
tags:
- Paper Reading
- Relation Classification
---
这是一篇ACL Findings的论文，idea很简单，但却非常奏效。

<!--more-->

关系分类旨在预测句子中两个实体间的关系，这篇论文通过实体类型来限制关系的搜索范围。例如两个实体类型都是`person`，那么他们的关系就可以排除`出生地`，这样就能减少候选关系的数量：

{% asset_img example.jpg %}

整体的模型结构如下（个人感觉画的不是很清晰）：

{% asset_img model.jpg %}

算法流程如下：

{% asset_img algorithm.jpg %}