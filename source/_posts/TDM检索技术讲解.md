---
title: TDM检索技术讲解
mathjax: true
toc: true
date: 2024-02-27 01:46:38
categories:
- 搜广推
tags:
- 召回
- Tree-based Model
---
召回的任务是从海量商品库中挑选出与用户最相关的topK个商品。传统的召回检索时间复杂度是$O(N)$，而阿里的TDM通过对全库商品构建一个树索引，将时间复杂度降低到$O(logN)$。

<!--more-->

## 模型概览

![model](https://raw.githubusercontent.com/TransformersWsz/picx-images-hosting/master/image.1zhztkufxh.webp)

树的每个节点输入到左侧复杂模型的时候，都是一个embedding，这样user向量和item向量可以提早交互，提升模型表达能力。