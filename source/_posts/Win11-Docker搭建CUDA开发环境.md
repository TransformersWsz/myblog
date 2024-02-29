---
title: Win11+Docker搭建CUDA开发环境
mathjax: true
toc: true
date: 2023-12-24 17:18:44
updated: 2023-12-24 17:18:44
categories:
- Machine Learning
tags:
- CUDA
- Docker
- Win11
---
最近入门了CUDA编程，先记录下搭建环境过程。

<!--more-->

由于在windows和wsl上折腾了好久，装cuda、cudnn、cmake、gcc等软件，还经常遇到依赖、版本许多问题，最终污染了系统环境。在朋友的安利下，采用docker容器开发方案，试一下真香。

## 本人软硬件条件
- OS: win11
- GPU: RTX 3060
- Driver Version: 537.42
- CUDA Version: 12.2
- Docker: Dokcer Desktop 4.12.0

目前想在docker容器里调用windows gpu，已经不再需要安装镜像nvidia-docker了。新版docker已经支持透传gpu，直接在参数里添加 `--gpus all` 即可：

```shell
docker run -it --gpus all --name gpu_test -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e NVIDIA_VISIBLE_DEVICES=all mortals/codeenv:conda-cuda11.8
```

Dockerfile见：[base-cuda11.8](https://github.com/mortals-debuging/pytorch-docker/blob/master/cuda11_8/base-cuda11.8.Dockerfile)

___

## 参考
- [2023完整版：深度学习环境在Docker上搭建（基于Linux和WSL）](https://zhuanlan.zhihu.com/p/646152162)