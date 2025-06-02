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
2. 偏态分布：有转化的人群中，LTV的非零值分布通常呈现出右偏重尾（分布的右侧有更长的尾巴，且均值 > 中位数 > 众数），即呈对数正态分布（Log-Normal）

<!--more-->

![data distribution](https://github.com/TransformersWsz/picx-images-hosting/raw/master/image.5fktxue58n.webp)

针对这种数据分布，Google提出了ZILN Loss，用于更真实地拟合这类零膨胀、长尾的数据。

___

LTV建模如下两个任务：用户是否付费、付多少费，分别对应上述两个问题。

$$
pred_ltv(x) = pay_{prob}(x) \times pay_{amount}(x)
$$

问题1是个二分类任务，问题2则是个回归任务：

$$
\begin{aligned}
    L_{\text {ZILN }}(x ; p, \mu, \sigma) &= L_{\text {CrossEntropy }}\left(\mathbb{1}_{\{x>0\}} ; p\right)+\mathbb{1}_{\{x>0\}} L_{\text {Lognormal }}(x ; \mu, \sigma) \\
    L_{\text {Lognormal }}(x ; \mu, \sigma) &= \log (x \sigma \sqrt{2 \pi})+\frac{(\log x-\mu)^2}{2 \sigma^2}
\end{aligned}

$$

