---
title: 'SGM: Sequence Generation Model for Multi-Label Classification'
mathjax: true
toc: true
date: 2025-03-09 02:44:40
updated: 2025-03-09 02:44:40
categories:
- NLP
tags:
- Multi-label Classification
- LSTM
---
为了建模多标签之间的依赖关系，本篇工作用序列生成的方式来解决该问题。

<!--more-->

## 模型结构

经典的序列生成范式：
$$
p(\boldsymbol{y} \mid \boldsymbol{x})=\prod_{i=1}^n p\left(y_i \mid y_1, y_2, \cdots, y_{i-1}, \boldsymbol{x}\right)
$$

![model](https://github.com/TransformersWsz/picx-images-hosting/raw/master/image.1e8r4ljlun.webp)