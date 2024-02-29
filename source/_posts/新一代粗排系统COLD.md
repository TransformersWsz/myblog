---
title: 新一代粗排系统COLD
mathjax: true
toc: true
date: 2024-02-21 01:37:18
updated: 2024-02-21 01:37:18
categories:
- 搜广推
tags:
- 粗排
- 交叉特征
---

![cold](https://raw.githubusercontent.com/TransformersWsz/picx-images-hosting/master//image.2vvvk4167lw0.webp)

为了让粗排支持交叉特征来提升模型性能，同时又为了降低引入交叉特征、复杂模型所带来的预估延迟和资源消耗，阿里团队提出了COLD，在模型效果和算力间取得了平衡。

<!--more-->

## 模型层面优化
1. 引入SENet，筛选出重要特征

## 算力层面优化
1. 并行拿特征
2. 列式计算：对于不同广告，它们的某些特征可能是相同，对列计算进行优化
3. 优化组合特征算子
4. fp16加速

## 吐槽
更多的是一篇工程优化文章，为了提升业务指标，对系统性能进行了各方面极致优化，叠加了不少硬件资源上去。但却给人一种大杂烩的感觉，没有从模型层面来优雅地创新。

___

## 参考
- [阿里定向广告最新突破：面向下一代的粗排排序系统COLD](https://zhuanlan.zhihu.com/p/186320100)