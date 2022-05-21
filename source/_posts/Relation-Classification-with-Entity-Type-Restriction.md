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

## 模型结构

{% asset_img model.jpg %}

## 算法流程

{% asset_img algorithm.jpg %}

$$
\begin{aligned}
R_{(t s, t o)} &=\left\{r \in R \mid(s, o) \in D_{r}\right\} \\
&=\{r \in R \mid t s \in S(r) \text { and } \text { to } \in O(r)\}
\end{aligned}
$$

1. 首先根据句子中实体的类型将句子归好组
2. 对于每一组，收集所有关系组成一个集合 $R_{(t s, t o)}$
3. 针对该关系集合学习一个分类器 $f_g$

## 实验结果
由于该方法是模型无关的，所以作者在两个代表性模型上做了实验：

{% asset_img result.png %}

`RECENT` 分别比 `GCN` 和 `SpanBERT` 高了6.9和4.4，提升还是非常明显的。

___
## Cite
```bib
@inproceedings{lyu-chen-2021-relation,
    title = "Relation Classification with Entity Type Restriction",
    author = "Lyu, Shengfei and Chen, Huanhuan",
    booktitle = "Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-acl.34",
    doi = "10.18653/v1/2021.findings-acl.34",
    pages = "390--395",
}
```