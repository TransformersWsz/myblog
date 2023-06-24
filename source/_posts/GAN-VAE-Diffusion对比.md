---
title: 'GAN,VAE,Diffusion对比'
mathjax: true
toc: true
date: 2023-06-24 17:11:47
categories:
- Machine Learning
tags:
- Image Generation
---

对比下三种主流图片生成模型的优缺点：

<!--more-->

## GAN

#### 优点
- 生成的图片逼真

#### 缺点
- 由于要同时训练判别器和生成器这两个网络，训练不稳定
- GAN主要优化目标是使图片逼真，导致图片多样性不足
- GAN的生成是隐式的，由网络完成，不遵循概率分布，可解释性不强


## VAE

#### 优点
- 学习的概率分布，可解释性强，图片多样性足

#### 缺点
- 产生图片模糊

## Diffusion
- 生成的图片逼真
- 数学可解释性强

#### 缺点
- 训练成本高昂、速度慢，需要多步采样

___

## 参考
- [DALLE2](https://www.bilibili.com/video/BV17r4y1u77B/?spm_id_from=333.999.0.0&vd_source=3f2411263f367ccf993c28b58688c0e7)
  - 该视频讲到了GAN、VAE、DVAE、VQ-VAE、Diffusion、DDPM、Improved DDPM、classifier-(free) guidance
- [从VAE到扩散模型：一文解读以文生图新范式](https://www.51cto.com/article/709837.html)