---
title: Enhancing CTR Prediction with De-correlated Expert Networks
mathjax: true
toc: true
date: 2025-08-06 02:40:16
updated: 2025-08-06 02:40:16
categories:
- 搜广推
tags:
- CTR
- MoE
---

本文探索了专家网络的差异性对模型性能的影响，本质上是种bagging思想，从各个语义空间上提升模型的表达能力。

<!--more-->

## 研究背景
- **核心问题**：MoE（混合专家）模型在CTR预估中，专家网络（Expert）的多样性对效果的影响
- **关键发现**：专家间差异性（不相关度）与模型AUC正相关  
  

## 方法论（Hetero-MoE）
#### 整体架构

![model](https://github.com/TransformersWsz/picx-images-hosting/raw/master/image.7zqqtylbd2.webp)

- **异构专家**：每个Expert使用独立Embedding + 不同网络结构（CrossNet/CIN/DNN等）
- **个性化Gate**：基于对应Expert的Embedding生成权重  
  

#### 差异性增强手段
| 维度         | 实现方式                          | 技术细节                                                                 |
|--------------|----------------------------------|--------------------------------------------------------------------------|
| **Embedding** | 每个Expert独立Embedding表        | 避免参数共享导致的表征同质化                                              |
| **结构异构**  | 混合CrossNet/CIN/DNN等不同结构    | 不同结构捕获多样特征交互模式                                              |
| **正则化**    | 皮尔逊相关系数损失                | $L_corr = ∑(Pearson(E_i, E_j))$，$E_i$为Expert i的输出向量                 |

## 实验效果
- **基准对比**：Hetero-MoE vs 传统MoE  
| 模型          | AUC提升 | 参数量  |
|---------------|---------|--------|
| Shared-MoE    | +0.0%   | 100%   |
| Hetero-MoE    | +1.8%   | 105%   |
  
- **消融实验**：  
  - 仅结构异构：+0.6% AUC  
  - 仅Embedding独立：+0.9% AUC  
  - 全方案：+1.8% AUC

## 关键公式
专家相关性损失（最小化皮尔逊系数）：
$$
\mathcal{L}_{corr} = \sum_{i \neq j} \left| \frac{\text{Cov}(E_i, E_j)}{\sigma_{E_i} \sigma_{E_j}} \right|
$$


---

## 参考

- [Enhancing CTR Prediction with De-correlated Expert Networks](https://arxiv.org/pdf/2505.17925)
- [中科大&腾讯：通过提升各个专家网络差异性提升基于MoE的CTR预估效果](https://mp.weixin.qq.com/s/JcvMQ5xJYLsCCWNqrv-ZiQ)
