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

使用常量内存的方法是在核函数外面用 `__constant__` 定义变量，并用函数 `cudaMemcpyToSymbol` 将数据从主机端复制到设备的常量内存后供核函数使用。

#### 纹理内存和表面内存
纹理内存和表面内存类似于常量内存，也是一种具有缓存的全局内存，有相同的可见范围和生命周期，而且一般仅可读(表面内存也可写)。不同的是，纹理内存和表面内存容量更大，而且使用方式和常量内存也不一样。

#### 寄存器
寄存器是一个线程能独立访问的资源，它所在的位置与局部内存不一样，是在片上（on chip）的存储，用来存储当前线程的一些暂存数据。寄存器的速度是访问中最快的，但是它的容量较小。

在核函数中定义的不加任何限定符的变量一般来说就存放于寄存器(register)中。各种内建变量，如 `gridDim、blockDim、blockIdx、 threadIdx 及 warpSize` 都保存在特殊的寄存器中，以便高效访问。举例如下：

```c
const int n = blockDim.x * blockIdx.x + threadIdx.x;
c[n] = a[n] + b[n];
```

`n` 也是一个寄存器变量，当只能被当前线程访问。

#### 局部内存
局部内存和寄存器几乎一样，核函数中定义的不加任何限定符的变量有可能在寄存器中，也有可能在局部内存中。寄存器中放不下的变量，以及索引值不能在编译时就确定的数组，都有可能放在局部内存中。

虽然局部内存在用法上类似于寄存器，但从硬件来看，局部内存只是全局内存的一部分。所以，局部内存的延迟也很高。每个线程最多能使用高达512KB的局部内存，但使用过多会降低程序的性能。

![detail](https://raw.githubusercontent.com/TransformersWsz/picx-images-hosting/master/image.b8p2wws76.webp)

___

## 转载
- [CUDA（二）：GPU的内存体系及其优化指南](https://zhuanlan.zhihu.com/p/654027980)