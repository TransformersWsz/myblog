---
title: 现代GPU内存分级结构
mathjax: true
toc: true
date: 2024-04-22 00:43:35
updated: 2024-04-22 00:43:35
categories:
- Machine Learning
tags:
- CUDA
- GPU
---
要实现CUDA高性能编程，就必须对GPU内存结构有深刻的了解。

<!--more-->

![GPU](https://raw.githubusercontent.com/TransformersWsz/picx-images-hosting/master/image.5tqtj239zp.png)

#### 全局内存
就是我们常说的显存，其容量最大、带宽最小、延迟最高。

#### 常量内存
存储在片下存储的设备内存上，但是通过特殊的常量内存缓存进行缓存读取，常量内存为只读内存，只有64KB。由于有缓存，常量内存的访问速度比全局内存高。

一个使用常量内存的方法是在核函数外面用 __constant__ 定义变量，并用 API 函数 cudaMemcpyToSymbol 将数据从主机端复制到设备的常量内存后 供核函数使用

![detail](https://raw.githubusercontent.com/TransformersWsz/picx-images-hosting/master/image.b8p2wws76.webp)


___

## 参考
- [CUDA（二）：GPU的内存体系及其优化指南](https://zhuanlan.zhihu.com/p/654027980)