---
title: metapath2vec解读
mathjax: true
toc: true
date: 2024-02-04 19:25:26
updated: 2024-02-04 19:25:26
categories:
- Machine Learning
tags:
- Heterogeneous Networks
- Paper Reading
---

metapath2vec在用在工业界的召回通路中比较多，非常适用于**异构的K部图**。

元路径 $P$ 定义形式如： $V_1 \rightarrow^{R_1} V_2 \rightarrow^{R_2} A_3 \ldots \rightarrow^{R_l} A_{l+1}$ 表示了从 $A_1$ 到 $A_{l+1}$ 的复杂关系。
其中 $V_i$ 表示节点类型，$R_i$ 表示节点间的关系。 $R=R_1 \circ R_2 \circ R_3 \circ R_l$，元路径 $P$ 的长度即为关系 $R$ 的个数。

<!--more-->
## 示例

`APA` 就表示两位作者是论文的共同作者：

![example](https://raw.githubusercontent.com/TransformersWsz/picx-images-hosting/a63833f2687f592ccad30d090fab8a88ce76a9c0/image.2squ8k783uy0.png)

## 不同类型的节点间的转移概率
$$
p\left(v^{i+1} \mid v_t^i, \mathcal{P}\right)=\left\{\begin{array}{cl}
\frac{1}{\left|N_{t+1}\left(v_t^i\right)\right|} & \left(v^{i+1}, v_t^i\right) \in E, \phi\left(v^{i+1}\right)=t+1 \\
0 & \left(v^{i+1}, v_t^i\right) \in E, \phi\left(v^{i+1}\right) \neq t+1 \\
0 & \left(v^{i+1}, v_t^i\right) \notin E
\end{array}\right.
$$

其中 $v_t^i$ 表示step $i$ 的类型为$V_t$的节点， $N_{t+1}\left(v_t^i\right)$ 表示 $v_t^i$ 的类型为$V_{t+1}$的邻居节点集合
> 具体详细的讲解见：[PGL系列16：metapath2vec](https://aistudio.baidu.com/aistudio/projectdetail/1099287)

## Heterogeneous Skip-Gram

$$
p\left(c_t \mid v ; \theta\right)=\frac{e^{X_{c_t} \cdot X_v}}{\sum_{u \in V} e^{X_u \cdot X_v}}
$$

## Metapath2vec++框架

$$
p\left(c_t \mid v ; \theta\right)=\frac{e^{X_{c_t} \cdot X_v}}{\sum_{u t \in V_t} e^{X_{u_t} \cdot X_v}}
$$

