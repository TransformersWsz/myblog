---
title: IPW逆概率加权
mathjax: true
toc: true
date: 2025-04-08 01:23:11
updated: 2025-04-08 01:23:11
categories:
- Marketing
tags:
- IPW
---
IPW是个非常优雅的纠偏方法。下面介绍如何利用它来实现纠偏：

<!--more-->

uplift定义如下：
$$
\begin{aligned}
\tau(x) &= \mathbb{E}_{\mathbb{P}}(Y(1)-Y(0) \mid x) \\
\mu_1(x) &= \mathbb{E}_{\mathbb{P}}(Y \mid W=1, X=x) \\
 \mu_0(x) &= \mathbb{E}_{\mathbb{P}}(Y \mid W=0, X=x) \\

\tau(x) &= \mu_1(x) - \mu_0(x)
\end{aligned}
$$


ESN纠偏：
$$
\begin{aligned}
\underbrace{P(Y, W=1 \mid X)}_{E S T R} & =\underbrace{P(Y \mid W=1, X)}_{T R} \cdot \underbrace{P(W=1 \mid X)}_\pi \\
& =\mu_1 \cdot \pi \\
\underbrace{P(Y, W=0 \mid X)}_{E S C R} & =\underbrace{P(Y \mid W=0, X)}_{C R} \cdot \underbrace{P(W=0 \mid X)}_{1-\pi} \\
& =\mu_0 \cdot(1-\pi) \\

\mu_1 &= \frac{ESTR}{\pi} \\
\mu_0 &= \frac{ESCR}{1-\pi} \\
\tau &= \mu_1 - \mu_0
\end{aligned}
$$

___

## 参考
- [DESCN: Deep Entire Space Cross Networks | 多任务、端到端、IPW的共舞](https://zhuanlan.zhihu.com/p/629853695)