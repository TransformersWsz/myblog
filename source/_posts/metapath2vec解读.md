---
title: metapath2vec解读
mathjax: true
toc: true
date: 2023-02-12 17:25:26
categories:
- Machine Learning
tags:
- Heterogeneous Networks
- Paper Reading
---
metapath2vec在用在工业界的召回通路中比较多。元路径 $P$ 是定义在网络模式 $TG = (A, R)$ 上的 $A_1 \rightarrow^R A_2 \rightarrow^R A_3 \ldots \rightarrow^R A_{l+1}$ 表示了从 $A_1$ 到 $A_{l+1}$ 的复杂关系。$R=R_1 \circ R_2 \circ R_3 \circ R_l$，元路径 $P$ 的长度即为关系 $R$ 的个数。

<!--more-->

> 具体详细的讲解见：[PGL系列16：metapath2vec](https://aistudio.baidu.com/aistudio/projectdetail/1099287)

## Heterogeneous Skip-Gram

$$
p\left(c_t \mid v ; \theta\right)=\frac{e^{X_{c_t} \cdot X_v}}{\sum_{u \in V} e^{X_u \cdot X_v}}
$$

## Metapath2vec++框架

$$
p\left(c_t \mid v ; \theta\right)=\frac{e^{X_{c_t} \cdot X_v}}{\sum_{u t \in V_t} e^{X_{u_t} \cdot X_v}}
$$

