---
title: GPU利用率
mathjax: true
toc: true
date: 2024-05-19 14:21:32
updated: 2024-05-19 14:21:32
categories:
- Machine Learning
tags:
- GPU
---
英伟达官方的GPU利用率的定义如下：
$$
GPU Util rate = \frac{number \  of \ active \ SM}{number \ of \ total \ SM} \times 100\%
$$

<!--more-->

## `nvidia-smi` 中的GPU利用率

```cpp
#include <stdio.h>

__global__ void simple_kernel() {
    while (true) {}
}

int main() {
    simple_kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
}
```

上述代码片段将在单个流多处理器(SM)上启动指定的内核(线程)。根据常规理解，GPU的“利用率”应该计算为$\frac{1}{num\_sm}$。但 `nvidia-smi` 却显示GPU利用率为100%：

![nvidia-smi](https://raw.githubusercontent.com/TransformersWsz/picx-images-hosting/master/image.45hhpwcfe7.webp)

根据NVML的定义，“利用率”是指在过去的样本期间内发生某些活动的时间百分比。具体来说：

- GPU利用率：这表示一个或多个内核在GPU上执行的时间百分比

NVML的定义完全不符合我们日常开发中的“利用率”理解。它仅测量给定采样周期内设备使用的时间部分，而不考虑该时间内使用的流式多处理器(SM)的数量。

通常，我们将“利用率”视为正在使用的GPU处理器的部分，用专业术语说就是“饱和度”：

> 资源具有无法服务的额外工作的程度

我们可以用 `dcgm-exporter` 来收集GPU的饱和度信息，这里引用[Tim在路上](https://mp.weixin.qq.com/s/4_An51JuRGWTU0dLgZYHpQ)的图片：

![gpu-util](https://raw.githubusercontent.com/TransformersWsz/picx-images-hosting/master/image.4g4bj2yxql.webp)

![sm](https://raw.githubusercontent.com/TransformersWsz/picx-images-hosting/master/image.7w6nb6993e.webp)


上图可以看到当GPU利用率为100%时，SM占用率非常低(<20%)，浮点运算(FP32/FP16/TensorCore)也保持在非常低的百分比，这表明GPU还没有饱和，而这才是真实的GPU利用现状。

___

## 参考
- [理解NVIDIA GPU 性能：利用率与饱和度](https://mp.weixin.qq.com/s/4_An51JuRGWTU0dLgZYHpQ)