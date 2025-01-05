---
title: 负采样后的CTR预估矫正
mathjax: true
toc: true
date: 2025-01-06 00:06:59
updated: 2025-01-06 00:06:59
categories:
- 搜广推
tags:
- Calibration
---
在搜广推场景中，正负样本不平衡是个普遍现象。通常做法是对负样本进行降采样，但采样后训练的模型预估概率会比实际概率高估。

<!--more-->

举例来说，线上真实样本的CTR是0.001，即正负样本比为1:1000。现对负样本降采样$w=0.01$，即采样后正负样本比为1:10，那么训练后的模型预估CTR为0.1，出现高估的情况。

预估率矫正推导如下：

- $N_{pos}$ 为线上正样本个数
- $N_{neg}$ 为线上负样本个数
- $N_{neg\_down}$ 为降采样后的负样本个数
- $w$ 为降采样概率
- $q$ 为线上CTR
- $p$ 为模型预估CTR

$$
\begin{aligned}
    N_{neg\_down} &= N_{neg} * w \\
    p &= \frac{N_{neg\_down}}{N_{neg\_down} + N_{pos}} \\
    q &= \frac{N_{neg}}{N_{neg}+N_{pos}} \\
\end{aligned}
$$