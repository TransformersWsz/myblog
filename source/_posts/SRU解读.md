---
title: SRU解读
mathjax: true
toc: true
date: 2023-06-15 02:30:21
updated: 2023-06-15 02:30:21
categories:
- NLP
tags:
- Paper Reading
- RNN
- Parallelizing
---
该篇论文实现了隐藏层维度的并行计算，但并没有解除时间步上的依赖。不过这样的改进，在模型训练和推理加速上的收益已经非常大了。

<!--more-->

笔记见：https://kdocs.cn/l/cbNfimpPLCvc
___

## 参考
- [Simple Recurrent Units了解一下](https://zhuanlan.zhihu.com/p/353500337)
- [如何评价新提出的RNN变种SRU?](https://www.zhihu.com/question/65244705)