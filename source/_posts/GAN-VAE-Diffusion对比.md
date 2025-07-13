---
title: 'GAN,VAE,Diffusion对比'
mathjax: true
toc: true
date: 2023-06-24 17:11:47
updated: 2023-06-24 17:11:47
categories:
- Machine Learning
tags:
- Image Generation
---

对比下三种主流图片生成模型的优缺点：

<!--more-->

| 模型      | 优点                         | 缺点 |
|:-------|:-------|:----|
| GAN   | <ul><li>生成的图片逼真</li></ul> |<ul><li>由于要同时训练判别器和生成器这两个网络，训练不稳定</li><li>GAN主要优化目标是使图片逼真，导致图片多样性不足</li><li>GAN的生成是隐式的，由网络完成，不遵循概率分布，可解释性不强</li></ul>|
| VAE   | <ul><li>学习的概率分布，可解释性强，图片多样性足</li></ul>  |<ul><li>图片生成质量差</li></ul>|
| Diffusion   | <ul><li>生成的图片逼真</li><li>数学可解释性强</li></ul>      |<ul><li>训练和推理成本高昂、速度慢，需要多步采样</li></ul>|


___

## 参考
- [DALL·E 2（内含扩散模型介绍）](https://www.bilibili.com/video/BV17r4y1u77B?t=1709.5)
  - 该视频讲到了GAN、VAE、DVAE、VQ-VAE、Diffusion、DDPM、Improved DDPM、classifier-(free) guidance
- [从VAE到扩散模型：一文解读以文生图新范式](https://www.51cto.com/article/709837.html)