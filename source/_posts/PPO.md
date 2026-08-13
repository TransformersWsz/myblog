---
title: PPO
mathjax: true
toc: true
date: 2026-08-13 22:50:14
updated: 2026-08-13 22:50:14
categories:
- Reinforcement Learning
tags:
- PPO
---

PPO(Proximal Policy Optimization)是近年来接触LLM始终绕不过去的RL算法，其核心思想如下：

> 根据经验调整Policy，但限制新旧Policy之间的变化幅度。

<!--more-->

## 算法背景

基于策略的方法包括PG和Actor-Critic算法，这些方法简单直观，但存在训练不稳定的情况。

假设现在 Policy 是：
```
左： 50%
右： 50%
```
经过一次训练，发现左边这个动作很好，于是梯度这样指引策略更新：
```
左： 50% → 90%
右： 50% → 10%
```
看起来没问题，但这次数据很可能只是偶然数据。到了下一次训练可能又发现，朝右动作也很好，那么：
```
左：90% → 10%
右：10% → 90%
```
如此策略就会疯狂震荡，无法收敛。为了解决这个问题，[TRPO](https://hrl.boyuai.com/chapter/2/trpo%E7%AE%97%E6%B3%95)应运而生，其通过复杂的数学约束，强行规定“新策略不能偏离老策略太远”，从而保证了稳定。但是TRPO的数学推导极其复杂，计算成本极高，很难在实际工程中落地。

## PPO的解法

PPO是TRPO的完美平替，它用一种极其简单巧妙的方法，实现了和TRPO一样的“限制更新幅度”的效果，而且代码实现简单、计算速度快。

PPO的核心目标函数如下：

$$
\begin{aligned}
    L^{CLIP}(\theta) &= \mathbb{E}_t \left[ \min \left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right] \\

    \theta^* &= \arg\max_{\theta} L^{CLIP}
\end{aligned}

$$


- $r_t(\theta)$：概率比
  - 公式：$\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$
  - 含义：`新策略采取这个动作的概率`除以`旧策略采取这个动作的概率`
  - 如果 $r>1$，意味着新策略比旧策略更倾向于做这个动作
- $\hat{A}_t$：优势函数，一般采用GAE计算
  - 公式：$\sum_{l=0}^{\infty}(\gamma\lambda)^l \delta_{t+l}$
  - 含义：评估该动作是否优于平均水平
  - 如果 $\hat{A}_t>0$，意味着说明在状态 $s$ 下采取动作 $a$，比状态 $s$ 下的平均动作水平要好，值得鼓励；如果 $\hat{A}_t<0$，说明该动作拖后腿了，要惩罚它
- $\text{clip}$：裁剪函数
  - 公式：$\text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)$
  - 含义：限制策略更新幅度，防止新旧策略差异过大，即“近端”的安全区域内
  - 设定 $\epsilon=0.2$，如果 $r>1.2$，裁剪函数将其强制拉回$1.2$；如果$r<0.8$，将其拉回$0.8$，从而保证了策略更新的稳定性

为了实现目标最大化，分如下两种情况：

> 当 $\hat{A}_t>0$，说明这个动作的价值高于平均，需要鼓励这个动作，则最大化 $r_t$，但不能超过 $1+\epsilon$
> 
> 当 $\hat{A}_t<0$，说明这个动作的价值不如平均，需要抑制这个动作，则最小化 $r_t$，但不能低于 $1-\epsilon$


## 总结

PPO是在Policy Gradient的基础上，通过比较新旧Policy的概率比例，并用Clip限制这个比例的变化范围，从而让 Policy能够朝着高Advantage的动作优化，同时避免一次更新过猛导致训练不稳定。
___

## 参考
- [第 12 章 PPO 算法](https://hrl.boyuai.com/chapter/2/ppo%E7%AE%97%E6%B3%95)

