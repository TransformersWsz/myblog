---
title: DIN解读
mathjax: true
toc: true
date: 2023-02-12 16:42:17
categories:
- Machine Learning
tags:
- CTR
- Recommender Systems
- Paper Reading
---
传统的Embedding&MLP架构将用户特征编码进一个固定长度的向量。当推出一个商品时，该架构无法捕捉用户丰富的历史行为中的多样性兴趣与该商品的关联。阿里妈妈团队提出了DIN网络进行改进，主要有如下两点创新：

<!--more-->

- 引入注意力机制来捕捉历史行为与当前商品的关联。用NMT的话来说，上文不同的单词对当前待生成的单词贡献不同，贡献高的应该赋予更大的权重，否则赋小
- 设计两种训练技巧来帮助训练大规模稀疏神经网络：
  - mini-batch aware正则化
  - 自适应激活函数




