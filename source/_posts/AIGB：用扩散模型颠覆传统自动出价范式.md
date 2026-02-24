---
title: AIGB：用扩散模型颠覆传统自动出价范式
mathjax: true
toc: true
date: 2026-02-24 23:24:30
updated: 2026-02-24 23:24:30
categories:
- Algorithm
tags:
- Diffusion
- Bidding
---

这是阿里巴巴在广告自动出价的一篇工作，用扩散模型颠覆了传统深度rl出价的范式，取得了线上线下的巨大收益。

<!--more-->
## 一、自动出价问题定义

在广告竞价中，广告主需要在有限预算下，为每个展示机会出价，最大化总价值：

$$
\begin{aligned}
\underset{o_i}{\text{maximize}} & \sum_i o_i v_i \\
\text{s.t.} & \sum_i o_i c_i \leq B \\
& \frac{\sum_i c_{ij}o_i}{\sum_i p_{ij}o_i} \leq C_j, \quad \forall j \\
& o_i \in \{0,1\}, \forall i
\end{aligned}
$$

理论上的最优出价形式为：

$$
b_i^* = \lambda_0 v_i + \sum_{j=1}^J \lambda_j p_{ij}
$$

其中 $\lambda_j$ 是需要动态调整的出价参数。

## 二、传统DRL出价 VS AIGB

**马尔可夫假设的问题**：传统RL假设下一状态只取决于当前状态和动作：
$$P(s_{t+1}|s_t,a_t,s_{t-1},a_{t-1},...) = P(s_{t+1}|s_t,a_t)$$

**但论文的统计分析发现**：随着历史序列长度增加，与下一状态的相关系数显著上升。

![correlation](https://raw.githubusercontent.com/TransformersWsz/picx-images-hosting/master/image.9rjxqqfgz6.webp)

这说明**历史信息**对预测未来状态很重要，但MDP假设丢弃了这些信息。


针对DRL的缺陷，AIGB直接建模总收益与整个状态轨迹的关联性：

| 方面 | 传统RL的MDP缺陷 | AIGB的全局建模如何解决 |
|------|-----------------|----------------------|
| **状态转移假设** | 只依赖当前状态 | 建模整个轨迹分布 |
| **长期依赖** | 误差累积 | 一次生成整个序列 |
| **稀疏回报** | 难以学习 | 直接以最终收益为条件 |
| **环境随机性** | 单步预测不稳 | 全局模式更鲁棒 |
| **约束满足** | 难控制 | 条件生成保证 |

## 三、AIGB范式：从"逐步决策"到"全局生成"

### 3.1 核心思想转变

| 维度 | 传统RL | AIGB（全局生成） |
|------|--------|------------------|
| **建模对象** | 单步转移 $P(s_{t+1}\|s_t,a_t)$ | 整个轨迹 $p(x_0(\tau)\|y(\tau))$ |
| **优化目标** | 最大化累计奖励 | 最大化条件似然 |
| **决策方式** | 逐步决策（online） | 全局规划后执行(Planning&Control) |

### 3.2 整体框架

![model](https://raw.githubusercontent.com/TransformersWsz/picx-images-hosting/master/image.2vfa6jytaj.webp)

1. **Planning生成整条轨迹**：用扩散模型生成整个未来状态轨迹
2. **Control生成出价动作**：用逆动力学模型反推出当前动作，逼近规划轨迹

## 四、DiffBid：扩散出价模型详解

### 4.1 问题建模

将自动出价建模为条件概率问题：

$$
\max_{\theta}\mathbb{E}_{\tau\sim D}[\log p_{\theta}(x_0(\tau)|\boldsymbol{y}(\tau))]
$$

其中：
- $x_0(\tau)$：完整状态轨迹 $[s_1, s_2, ..., s_T]$，$s_t$包含剩余预算、预算消耗速度等等
- $\boldsymbol{y}(\tau)$：轨迹属性，包含总收益、约束条件等等

### 4.2 扩散过程设计

#### 前向加噪
$$
q(x_k(\tau)|x_{k-1}(\tau)) = \mathcal{N}(x_k(\tau);\sqrt{1-\beta_k}x_{k-1}(\tau),\beta_k I)
$$


#### 反向去噪

1. 预测噪音：
$$
\hat{\epsilon}_k = \epsilon_{\theta}(x_k(\tau),k) + \omega (\epsilon_{\theta}(x_k(\tau),\boldsymbol{y}(\tau),k) - \epsilon_{\theta}(x_k(\tau),k))
$$
2. 去噪生成下一状态轨迹：
$$
\boldsymbol{x}_{k-1}(\tau) \sim \mathcal{N}\left(\boldsymbol{x}_{k-1}(\tau) \mid \boldsymbol{\mu}_\theta\left(x_k(\tau), \boldsymbol{y}(\tau), k\right), \Sigma_\theta\left(\boldsymbol{x}_k(\tau), k\right)\right)
$$


### 4.3 逆动力学：从未来状态反推动作

$$
\hat{\boldsymbol{a}}_t = f_{\phi}(s_{t-L:t}, s_{t+1}')
$$

根据历史状态和预测的下一个目标状态，直接生成当前应采取的最优出价动作，即$\lambda_0, \lambda_1, \dots,\lambda_J$

### 4.4 训练loss

$$
\mathcal{L}(\theta,\phi) = \mathbb{E}_{k,\tau \in \mathcal{D}}\left[||\epsilon -\epsilon_{\theta}(x_k(\tau),\boldsymbol{y}(\tau),k)||^2\right] + \mathbb{E}_{(s_{t-L:t},a_t,s_{t+1}')\in \mathcal{D}}\left[||\boldsymbol{a}_t - f_{\phi}(s_{t-L:t},s_{t+1}')||^2\right]
$$

### 4.5 线上推理流程

![inference](https://raw.githubusercontent.com/TransformersWsz/picx-images-hosting/master/image.60us5huy4x.webp)

- 每次时间步$t$**重新生成整个未来轨迹**，即$t-1$生成的$x_0(\tau)$与$t$生成的$x_0(\tau)$无关
- 根据历史状态和预测下一状态，用idm来生成出价动作
- 每步解码都注入历史状态保证已发生的不变


## 五、实验结果

![experiment](https://raw.githubusercontent.com/TransformersWsz/picx-images-hosting/master/image.8l0mi4wchh.webp)
DiffBid在各数据集上都取得了sota，并大幅领先所有baseline。


## 六、FAQ
1. AIGB是根据马尔科夫假设单步去噪的，它是如何体现全局建模的？

实际上这两者并不冲突，每步diffusion去噪是生成整个状态轨迹。而全局建模是指在单条状态轨迹中，所有历史状态均对最终收益产生直接影响，有如下两点体现：
- 建模MLE：$\max_{\theta}\mathbb{E}_{\tau\sim D}[\log p_{\theta}(x_0(\tau)|\boldsymbol{y}(\tau))]$
- 历史状态和预测状态影响出价动作：$\hat{\boldsymbol{a}}_t = f_{\phi}(s_{t-L:t}, s_{t+1}')$

2. diffusion需要多步去噪，线上RT高如何解决？
![online](https://raw.githubusercontent.com/TransformersWsz/picx-images-hosting/master/image.3rbrm0ep1e.webp)

论文里也提到了这个问题，推理耗时与去噪步数成正比。对于文生图模型，为确保图片质量，步数会非常大，但对于出价问题，较小的步数已经能保证较好的实验效果，且自动出价对实时性要求不高。
