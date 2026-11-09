---
title: 精排scaling调研
mathjax: true
toc: true
date: 2026-11-05 01:43:41
updated: 2026-11-05 01:43:41
categories:
- 搜广推
tags:
- Scaling
- Transformer
---

集中调研下Scaling在搜广推领域的进展。

<!--more-->

## 0. 背景：推荐精排为什么缺 Scaling Law

LLM 的公式是：**统一的 Transformer block + 堆参数堆数据 = 效果可预测地涨**。推荐精排模型做不到，卡点在输入异构：
