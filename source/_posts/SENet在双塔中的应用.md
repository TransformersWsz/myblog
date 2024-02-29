---
title: SENet在双塔中的应用
mathjax: true
toc: true
date: 2024-02-05 01:14:04
updated: 2024-02-05 01:14:04
categories:
- 搜广推
tags:
- Dual Tower
- SENet
- 召回负例
- 粗排负例
---

SENet思想非常简单，模型结构如下：

<!--more-->

![SENet](https://raw.githubusercontent.com/TransformersWsz/picx-images-hosting/96cf35a3f9d6db7affd5bca632a35891f42f9f91/image.4a55hw1xznq0.webp)

对比推荐系统，将CV图像的通道数 $channel_{num}$ 看作是user侧或ad侧的 $feature_{num}$ 。以user侧特征 $v_i$ 为例：

## Squeeze

$$
z_i = \frac{\sum_{t=1}^k v_i^t}{k}
$$
将特征 $v_i$ 的embeeding平均池化成 $z_i$，embedding size为 $k$

## Excitation

$$
S = \phi(W_2 \phi(W_1 Z)), Z \in R^{feature_{num}}
$$

- $W_1$ 用于降维，从而实现将池化后的特征进行特征交互，生成中间向量
- $\phi$ 是激活函数
- $W_2$ 用于升维，将中间向量恢复成 $feature_{num}$ 个权重值，得到每个特征的重要性

## 总结
SENet并不能将user侧和item侧的特征交互提前或者使其获得更深层次的交互，user侧和item侧的特征交互仍然只发生在最后的内积那一步，这是由其双塔结构导致的。SENet的作用是提前将各侧的重要特征升权，不重要特征降权。

## 个人疑问
既然SENet目的是为了对特征进行升降权，那么为什么不直接在特征侧引入一个可学习的参数矩阵呢？
___

## 参考
- [SENet双塔模型：在推荐领域召回粗排的应用及其它](https://zhuanlan.zhihu.com/p/358779957)
- [解读Squeeze-and-Excitation Networks（SENet）](https://zhuanlan.zhihu.com/p/32702350)