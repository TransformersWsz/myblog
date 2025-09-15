---
title: 图像生成评估指标IS&FID
mathjax: true
toc: true
date: 2025-09-18 00:02:42
updated: 2025-09-18 00:02:42
categories:
- CV
tags:
- Image Generation
---

**IS（Inception Score）** 和 **FID（Fréchet Inception Distance）** 是评估生成模型（特别是GAN、Diffusion等）最常见的两个指标

<!--more-->

## Inception Score (IS)
### 思想

* 希望生成的图片**清晰且多样**。
* 用预训练好的**Inception v3 分类网络**来评估生成图片的质量。

### 公式

对生成图片$x$，Inception网络输出类别分布$p(y|x)$。

* **清晰度**：若图片清晰，$p(y|x)$ 应该高度集中（熵低）。
* **多样性**：若生成结果多样，整体类别分布 $p(y) = \int_x p(y|x) dx$ 应该接近均匀（熵高）。

定义 IS：

$$
IS = \exp\left(\mathbb{E}_{x} \left[ D_{KL}(p(y|x) \parallel p(y)) \right]\right)
$$

### 特点

* **优点**：简单直观，广泛使用。
* **缺点**：

  1. 依赖 Inception v3 分类器，不一定适用于非 ImageNet 数据集。
  2. 只看类别分布，不直接衡量“真实分布的接近度”。

## Fréchet Inception Distance (FID)

### 思想

* 用统计方式比较**真实图片分布**与**生成图片分布**的接近程度。
* 在Inception v3特征空间里，把数据分布近似成高斯分布，然后计算两者的Fréchet 距离（Wasserstein-2 距离）。

### 公式

设真实图片特征的分布为 $\mathcal{N}(\mu_r, \Sigma_r)$，生成图片特征的分布为$\mathcal{N}(\mu_g, \Sigma_g)$。
FID 定义为：

$$
FID = \|\mu_r - \mu_g\|^2 + \mathrm{Tr}\left(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2}\right)
$$

### 特点

* **优点**：

  1. 能综合反映图像的质量和多样性。
  2. 与人类感知一致性更高。
  3. 可以比较不同数据集。
* **缺点**：

  1. 特征分布假设为高斯，近似可能不准。
  2. 计算时需要足够样本，否则估计不稳定。


## 对比总结

| 指标      | 思想                   | 优点                 | 缺点                          |
| ------- | -------------------- | ------------------ | --------------------------- |
| **IS**  | 用 KL 散度衡量单图置信度与整体多样性 | 简单、计算快             | 依赖 Inception，不能直接衡量与真实分布的差距 |
| **FID** | 在特征空间拟合高斯，计算两分布差异    | 更符合人类感知，能比较生成与真实数据 | 需要更多样本，假设近似可能偏差             |


👉 直观理解：

* **IS 高 → 图像清晰且类别多样**
* **FID 低 → 生成分布接近真实分布**

