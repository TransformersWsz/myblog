---
title: GRPO算法
mathjax: true
toc: true
date: 2026-10-01 20:12:21
updated: 2026-10-01 20:12:21
categories:
- Reinforcement Learning
tags:
- PPO
- GRPO
---

GRPO（**Group Relative Policy Optimization**，分组相对策略优化）是 DeepSeek 团队在 **DeepSeek-Math** 和 **DeepSeek-R1** 中提出的强化学习算法，核心动机是**砍掉 Critic 模型**。

![grpo](https://github.com/TransformersWsz/picx-images-hosting/raw/master/image.9rk6cm06kk.png)

<!--more-->

## PPO 的痛点

回顾经典 RLHF 的 PPO：

```
┌──────────────────────────────────────┐
│  PPO 需要四个模型同时驻留显存：        │
│                                       │
│  ① Actor πθ      ~175B               │
│  ② Critic Vφ     ~175B  ← 砍掉它！    │
│  ③ Reference πref ~175B              │
│  ④ Reward rψ     ~175B               │
│                                       │
│  总显存 ≈ 4 × 175B = 700B 参数量     │
└──────────────────────────────────────┘
```

Critic 模型与 Actor 参数量级相同，训练时占用巨大显存，且 Critic 本身的价值估计误差可能引入训练不稳定性。

> GRPO 的核心思路：不需要 Critic，用同一 prompt 下的一组回答的相对好坏来估计优势。

## 建模方式

#### 1. 核心思想：组内相对比较

```
传统 PPO（有 Critic）：
  生成一个回答 → Critic 估计 V(s) → 算优势 A = r - V(s)

GRPO（无 Critic）：
  对同一个 prompt 生成 G 个回答 → RM 给每个打分 →
  组内归一化 → 相对分数即优势
```

#### 2. 数学形式

对于每个 prompt $q$，Actor 采样一组 $G$ 个回答 $\{o_1, o_2, ..., o_G\}$：

1. 用 RM 对每个回答打分，得到 $\{r_1, r_2, ..., r_G\}$

2. 组内归一化，计算每个回答的相对优势：

$$A_i = \frac{r_i - \text{mean}(r_1, ..., r_G)}{\text{std}(r_1, ..., r_G)}$$

这就是 **group relative** 的含义——每个回答的优势是它相对于同组其他回答的标准化分数。

3. GRPO 目标函数：

$$
J_{\mathrm{GRPO}}=\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}\min\Big(\rho_{i,t}\hat{A}_{i,t},\ \mathrm{clip}\big(\rho_{i,t},\,1-\epsilon,\,1+\epsilon\big)\hat{A}_{i,t}\Big)-\beta D_{\mathrm{KL}}\big(\pi_\theta\,\|\,\pi_{\mathrm{ref}}\big).
$$

- $\frac { \pi _ { \theta } ( o _ { i , t } | q , o _ { i , < t } ) } { \pi _ { \theta _ { o l d } } ( o _ { i , t } | q , o _ { i , < t } ) }$，==逐token计算==，新旧策略概率比，与PPO保持一致。
- $\hat{A}_{i,t}=A_i$，==逐response计算==，优势函数定义在整条 response 上。若第 $i$ 条回答优于同组平均，它从第一个 token 到最后一个 token 都乘上同一个正优势；同一 response 内的不同 token 因而无法区分。

#### 3. 与 PPO 的关键区别

| 维度 | PPO | GRPO |
|------|-----|------|
| 优势计算 | $A = r + \gamma V(s') - V(s)$ 需要 Critic | $A = \frac{r - \text{mean}(r)}{\text{std}(r)}$ 组内归一化 |
| 模型数量 | 4 个 | **3 个**（无 Critic） |
| 采样方式 | 1 个 prompt → 1 个回答（逐 token） | 1 个 prompt → **G 个完整回答** |
| KL 惩罚 | 逐 token 计算 | 逐response计算 |


## 优缺点分析

#### ✅ 优点

| 优点 | 说明 |
|------|------|
| **显存大幅减少** | 砍掉 Critic 模型，训练时少一个 ~175B 的模型，显存降低约 25-30% |
| **训练更稳定** | 消除了 Critic 价值估计不准确带来的方差，避免"Critic 带偏 Actor" |
| **实现更简单** | 不需要维护 Critic 网络、不需要 GAE 计算、不需要价值函数拟合 |
| **天然适合数学/代码推理** | 这类任务的奖励通常是规则化的（最终答案对/错），组内比较很自然 |
| **组内归一化自带正则** | 标准化操作天然控制了梯度尺度，减少超参敏感度 |

#### ❌ 缺点

| 缺点 | 说明 |
|------|------|
| **计算量增大** | 每个 prompt 要生成 G 个回答（通常 G=4~16），前向推理量 ×G |
| **依赖 RM 质量** | 没有 Critic 兜底，完全依赖 RM 打分，RM 的偏序错误会直接影响训练 |
| **只适用于有明确奖励的场景** | 只适用于数学、代码这类客观题；如果奖励是稀疏的、或难以规则化定义，组内比较可能不够精细 |
| **组大小 G 是敏感超参** | G 太小则方差大、归一化不稳定；G 太大则推理开销难以承受 |
| **无法做逐 token 的细粒度 credit assignment** | 优势是response级别的，不是 token 级别的，可能忽略过程中细节 |

## 总结

| 维度 | PPO（经典 RLHF） | GRPO（DeepSeek） |
|------|------------------|------------------|
| 模型数量 | 4（Actor + Critic + Ref + RM） | 3（Actor + Ref + RM） |
| 优势来源 | Critic 价值估计 | 组内相对比较 |
| 显存占用 | 大 | 小（省一个 Critic） |
| 推理开销 | 1× | G×（组大小倍数） |
| 稳定性 | 依赖 Critic 质量 | 依赖 RM + G |
| 代表工作 | InstructGPT, ChatGPT | DeepSeek-Math, DeepSeek-R1 |

> GRPO 用"同一 prompt 生成多个回答、组内互相比"替代了 Critic 的价值估计，省显存、更稳定、特别适合有明确对错标准的推理任务，代价是推理开销翻 G 倍。

___

## 参考
- [DeepSeekMath: Pushing the Limits of Mathematical
Reasoning in Open Language Models](https://arxiv.org/pdf/2402.03300)
- [GRPO：DeepSeek 的潘朵拉盒子](https://zhuanlan.zhihu.com/p/1984387073625593089)