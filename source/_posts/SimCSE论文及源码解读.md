---
title: SimCSE论文及源码解读
mathjax: true
toc: true
date: 2022-05-01 16:46:40
categories:
- NLP
tags:
- 论文阅读
- Dropout
- Contrastive Learning
---
对比学习的思想是拉近同类样本的距离，增大不同类样本的距离，目标是要从样本中学习到一个好的语义表示空间。SimCSE是一种简单的无监督对比学习框架，它通过对同一句子两次Dropout得到一对正样例，将该句子与同一个batch内的其它句子作为一对负样例。模型结构如下所示：

<!--more-->

![simcse](https://cdn.jsdelivr.net/gh/TransformersWsz/image_hosting@master/simcse.ldig50thwww.jpg)

损失函数为：
$$
\ell_{i}=-\log \frac{e^{\operatorname{sim}\left(\mathbf{h}_{i}^{z_{i}}, \mathbf{h}_{i}^{z_{i}^{\prime}}\right) / \tau}}{\sum_{j=1}^{N} e^{\operatorname{sim}\left(\mathbf{h}_{i}^{z_{i}}, \mathbf{h}_{j}^{z_{j}^{\prime}}\right) / \tau}}
$$

___

## 参考
- [“被玩坏了”的Dropout](https://mp.weixin.qq.com/s/IDWih5h2rLNqr3g0s8Y9zQ)
- [细节满满！理解对比学习和SimCSE，就看这6个知识点](https://mp.weixin.qq.com/s/12UvfXnaB4NTy54wWIFZdQ)
- [SIMCSE算法源码分析](https://zhuanlan.zhihu.com/p/483453992)