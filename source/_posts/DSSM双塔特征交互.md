---
title: DSSM双塔特征交互
mathjax: true
toc: true
date: 2024-07-09 01:00:42
updated: 2024-07-09 01:00:42
categories:
- 搜广推
tags:
- Feature Interaction
- Dual Tower
---
传统的DSSM双塔无法在早期进行user和item侧的特征交互，这在一定程度上降低了模型性能。我们想要对双塔模型进行细粒度的特征交互，同时又不失双塔模型离线建向量索引的解耦性。下面介绍两篇这方面的工作。
<!--more-->