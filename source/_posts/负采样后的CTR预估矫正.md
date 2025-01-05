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

举例来说，