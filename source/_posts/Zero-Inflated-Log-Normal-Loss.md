---
title: Zero-Inflated Log-Normal Loss
mathjax: true
toc: true
date: 2025-06-03 02:13:22
updated: 2025-06-03 02:13:22
categories:
- Marketing
tags:
- LTV
- ZILN
---

在营销LTV预测任务中，用户的价值呈现出如下特点：
1. 零膨胀（Zero-inflation）：大量用户的LTV为零（比如没有转化、没有付费）
2. 偏态分布：有转化的人群中，LTV的非零值分布通常呈现出右偏重尾，即呈对数正态分布（Log-Normal）。

![data distribution](https://github.com/TransformersWsz/picx-images-hosting/raw/master/image.5fktxue58n.webp)

<!--more-->

针对这种数据分布，Google提出了ZILN Loss，用于更真实地拟合这类零膨胀、长尾的数据。

___

LTV建模如下两个任务：

$$
$$