---
title: SENet在双塔中的应用
mathjax: true
toc: true
date: 2024-02-04 17:14:04
categories:
- 搜广推
tags:
- Dual Tower
- SENet
- 召回负例
- 粗排负例
---

以user侧特征为例， $v_i^{}$

## Squeeze

$$
z_i = \frac{\sum_{t=1}^k v_i^t}{k}
$$
将特征 $v_i$ 的embeeding平均池化成 $z_i$

## Excitation

$$
S = \phi(W_2 \phi(W_1 Z))
$$

- $W_1$ 用于将压缩后的特征进行特征交互
- $\phi$ 是激活函数
- $W_2$ 用于升维，将该特征维度恢复成embedding size

## 总结
ENet并不能将user侧和item侧的特征交互提前或者使其获得更深层次的交互，user侧和item侧的特征交互仍然只发生在最后的内积那一步，这是由其双塔结构导致的。SENet的作用是提前将各侧的重要特征升权，不重要特征降权。
___

## 参考
- [SENet双塔模型：在推荐领域召回粗排的应用及其它](https://zhuanlan.zhihu.com/p/358779957)