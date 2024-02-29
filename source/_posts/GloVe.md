---
title: GloVe
mathjax: true
toc: true
date: 2021-07-21 01:11:50
updated: 2021-07-21 01:11:50
categories:
- Algorithm
tags:
- Word2Vec
- 面试
---
GloVe的全称叫Global Vectors for Word Representation，它是一个基于全局词频统计（count-based & overall statistics）的词表征工具，它可以把一个单词表达成一个由实数组成的向量，这些向量捕捉到了单词之间一些语义特性，比如相似性、类比性等。

<!--more-->

## 构建共现矩阵
设共现矩阵为 $X$ ，其元素为 $X_{i,j}$ 。

$X_{i,j}$ 的意义为：在整个语料库中，单词 $i$ 和单词 $j$ 共同出现在一个窗口中的次数。

具体示例见：https://blog.csdn.net/coderTC/article/details/73864097

## 词向量与共现矩阵的近似关系
构建词向量（Word Vector）和共现矩阵（Co-ocurrence Matrix）之间的近似关系，论文的作者提出以下的公式可以近似地表达两者之间的关系：

$$
\log X_{i k}=w_{i}^{T} w_{k}+b_{i}+b_{k}
$$

具体公式推导见：https://zhuanlan.zhihu.com/p/42073620

## 构造损失函数
$$
J=\sum_{i k} f\left(X_{i k}\right)\left(w_{i}^{T} w_{k}+b_{i}+b_{k}-\log X_{i k}\right)^{2}
$$

$f(x)$ 为权重函数，满足如下三个特点：
- $f(0)=0$ ，即两个单词没有在同一个滑动窗口中出现过，那么它们不应该参与到loss的计算中；
- $f(x)$ 为非递减函数，即这些单词的权重要大于那些很少在一起出现的单词；
- $f(x)$ 不能过大，达到一定程度后不再增加。如果汉语中“这”出现很多次，但重要程度很小；

综上 $f(x)$ 定义如下：
$$
f(x)=\left\{\begin{array}{c}
\left(\frac{x}{x_{\max }}\right)^{\alpha}, \text { if } x<x_{\max } \\
1, \text { otherwise }
\end{array}\right.
$$

## GloVe与LSA、Word2Vec的区别
- LSA是基于奇异值分解（SVD）的算法，该方法对term-document矩阵（矩阵的每个元素为tf-idf）进行奇异值分解，从而得到term的向量表示和document的向量表示。此处使用的tf-idf主要还是term的全局统计特征。而我们SVD的复杂度是很高的，所以它的计算代价比较大。还有一点是它对所有单词的统计权重都是一致的。
- word2vec最大的缺点则是没有充分利用所有的语料，只利用了局部的上下文特征。
- GloVe模型就是将这两种特征合并到一起的，即使用了语料库的全局统计（overall statistics）特征，也使用了局部的上下文特征（即滑动窗口）。

___

## 参考

- [理解GloVe模型（Global vectors for word representation）](https://blog.csdn.net/coderTC/article/details/73864097)
- [（十五）通俗易懂理解——Glove算法原理](https://zhuanlan.zhihu.com/p/42073620)
- [ML-NLP](https://github.com/NLP-LOVE/ML-NLP)