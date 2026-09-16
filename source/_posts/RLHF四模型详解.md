---
title: RLHF四模型详解
mathjax: true
toc: true
date: 2026-09-17 01:47:49
updated: 2026-09-17 01:47:49
categories:
- NLP
tags:
- LLM
- PPO
- RM
- Actor-Critic
---

在 RLHF（Reinforcement Learning from Human Feedback）的 PPO 训练阶段，一共涉及四个模型：**Actor、Critic、Reference、Reward**。它们各司其职，协作完成“让语言模型输出更符合人类偏好”的目标。

本文将从每个模型的作用出发，详细推导各自的Loss，最后给出总 Loss的完整形式。



## 一、四个模型分别是什么？

| 模型 | 是否训练 | 作用 | 输出 |
|------|---------|------|------|
| **Actor** $\pi_\theta$ | ✅ 训练 | 生成回答，被优化 | 每个 token 的概率 |
| **Critic** $V_\phi$ | ✅ 训练 | 估计状态价值，算优势 | 标量 $V(s)$ |
| **Reference** $\pi_{ref}$ | ❌ 冻结 | 提供 KL 基准，防跑偏 | 每个 token 的概率 |
| **Reward** $r_\psi$ | ❌ 冻结 | 对完整回答打分 | 标量 $r$ |


### 1. Actor（策略模型）

- 就是我们要训练的语言模型本身，通常从 SFT 模型初始化。
- 给定 prompt，自回归生成回答。
- 通过 PPO 更新参数，使得生成的回答获得更高 reward，同时不偏离 Reference 太远。

### 2. Critic（价值模型）

- 一个与 Actor 规模相近的模型，输出一个标量价值 $V(s)$。
- 估计当前状态（已生成的 token 序列）的“预期回报”，用于计算优势函数。
- 只在训练时使用，推理时不参与。

### 3. Reference（参考模型）

- 通常是冻结的 SFT 模型（Actor 的初始副本），参数不更新。
- 提供“原始行为”的基准，用来计算 KL 散度惩罚。
- 防止 Actor 为了刷高 reward 而过度偏离原始语言模型，导致输出退化。

### 4. Reward（奖励模型）

- 用人类偏好数据训练出来的打分模型，输出一个标量奖励。
- 对 Actor 生成的完整回答打分，衡量它有多符合人类偏好。
- 参数冻结，只做推理打分。


## 二、训练流程概览

以一条训练样本为例：

1. **Actor** 根据 prompt $x$ 生成回答 $y = (y_1, ..., y_n)$
2. **Reward** 对完整回答打分 $r(x,y)$
3. **Reference** 对同一回答计算 log 概率，用于 KL 惩罚
4. 计算每步奖励 $R_t$
5. **Critic** 估计每个状态的价值 $V_t$
6. 用 GAE 计算优势 $\hat{A}_t$
7. 用 PPO 更新 Actor 和 Critic


## 三、设定与符号

- 输入 prompt：$x = (x_1, ..., x_m)$
- Actor 生成的输出：$y = (y_1, ..., y_n)$
- 每个 $y_t$ 是一个 token

四个模型的前向输出：

| 模型 | 输入 | 输出 |
|------|------|------|
| Actor $\pi_\theta$ | $x, y_{<t}$ | $\log \pi_\theta(y_t \mid x, y_{<t})$ |
| Reference $\pi_{ref}$ | $x, y_{<t}$ | $\log \pi_{ref}(y_t \mid x, y_{<t})$ |
| Reward $r_\psi$ | $x, y$ | 标量 $r$ |
| Critic $V_\phi$ | $x, y_{<t}$ | 标量 $V_t$ |


## 四、每步奖励 $R_t$ 的计算

### 4.1 KL 惩罚

逐 token 的 log 概率差，衡量 Actor 相对 Reference 偏离了多少：
$$
\text{KL}_t = \log \pi_\theta(y_t \mid x, y_{<t}) - \log \pi_{ref}(y_t \mid x, y_{<t})
$$



### 4.2 每步奖励

$$
R_t = -\beta \cdot \text{KL}_t, \quad t = 1, ..., n-1
$$

$$
R_n = -\beta \cdot \text{KL}_n + r
$$

- 中间 token：只有 KL 惩罚
- 最后一个 token：KL 惩罚 + Reward 模型的分数

**直观**：
- Actor 和 Reference 概率一样时，$\text{KL}_t = 0$，无惩罚
- Actor 偏离越大，$\text{KL}_t$ 越大，$R_t$ 越负
- 最后一步加上 $r$，注入人类偏好信号


## 五、Critic 的 Loss

### 5.1 TD 残差

$$
\delta_t = R_t + \gamma V_{t+1} - V_t
$$

其中 $V_{n+1} = 0$。

- $R_t$：这一步实际拿到的奖励
- $\gamma V_{t+1}$：对下一步价值的折扣估计
- $V_t$：对当前价值的估计
- 差值 $\delta_t$：实际比预期好多少

### 5.2 优势函数 GAE

从后往前递推：

$$
\hat{A}_n = \delta_n
$$

$$
\hat{A}_t = \delta_t + \gamma \lambda \hat{A}_{t+1}, \quad t = n-1, ..., 1
$$

展开形式：

$$
\hat{A}_t = \sum_{l=0}^{n-t} (\gamma \lambda)^l \delta_{t+l}
$$

> **含义**：第 $t$ 个 token 的优势 = 当前 TD 误差 + 未来 TD 误差的折扣加权和。

### 5.3 Critic 的 target

$$
\hat{V}_t^{target} = \hat{A}_t + V_{\phi_{old}}(s_t)
$$


### 5.4 Critic 的 Loss

$$
L^{critic} = \frac{1}{n} \sum_{t=1}^{n} \left( V_\phi(x, y_{<t}) - \hat{V}_t^{target} \right)^2
$$

- $V_\phi(x, y_{<t})$：当前 Critic 预测，参与梯度。
- $\hat{V}_t^{target}$：目标值，不参与梯度。
- 均方误差，让 Critic 逼近目标。

**直观**：Critic 学着预测“从这个前缀出发能拿多少分”，预测越准，优势估计越可靠。


## 六、Actor 的 Loss

### 6.1 重要性采样比

$$
\rho_t = \frac{\pi_\theta(y_t \mid x, y_{<t})}{\pi_{\theta_{old}}(y_t \mid x, y_{<t})}
$$

- $\pi_\theta$：当前策略（参与梯度）
- $\pi_{\theta_{old}}$：采样时的旧策略（不参与梯度）
- $\rho_t$：同一个 token，新旧策略概率的比值


### 6.2 裁剪项

$$
L^{clip}_t = \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon) \hat{A}_t
$$

其中 $\epsilon$ 通常取 0.1 或 0.2。

### 6.4 Actor 的 Loss

$$
L^{actor} = -\frac{1}{n} \sum_{t=1}^{n} \min\left( L^{unclip}_t, L^{clip}_t \right)
$$

前面加负号是因为我们要**最大化**这个目标，而优化器默认**最小化** loss。

### 6.5 Clip 机制逐情况分析

| 优势 | 概率变化 | clip 后行为 |
|------|---------|-----------|
| $\hat{A}_t > 0$ | 想提高 | 超过 $1+\epsilon$ 就不再奖励 |
| $\hat{A}_t < 0$ | 想降低 | 低于 $1-\epsilon$ 就不再惩罚 |
| $\rho_t$ 在范围内 | 正常更新 | clip 不起作用 |

> **一句话**：PPO 用 clip 把每次策略更新限制在一个“信任域”内，避免一步走太远导致训练不稳定。


## 七、熵奖励

$$
L^{entropy} = \frac{1}{n} \sum_{t=1}^{n} \sum_{v} \pi_\theta(v \mid x, y_{<t}) \log \pi_\theta(v \mid x, y_{<t})
$$

- 熵衡量策略的“随机性”或“探索程度”
- 加熵奖励防止策略过早收敛到局部最优、输出多样性下降
- 在 RLHF 中，熵奖励通常很小（$c_2 = 0.01$ 或更小）


## 八、总 Loss

$$
\boxed{
L^{PPO} = L^{actor} + c_1 L^{critic} - c_2 L^{entropy}
}
$$

展开形式：

$$
L^{PPO} =
\underbrace{-\frac{1}{n} \sum_{t=1}^{n} \min\left( \rho_t \hat{A}_t, \ \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon) \hat{A}_t \right)}_{Actor}
$$

$$
+ \ c_1 \underbrace{\frac{1}{n} \sum_{t=1}^{n} \left( V_\phi(s_t) - \hat{V}_t^{target} \right)^2}_{Critic}
$$

$$
- \ c_2 \underbrace{\frac{1}{n} \sum_{t=1}^{n} \sum_{v} \pi_\theta(v \mid s_t) \log \pi_\theta(v \mid s_t)}_{Entropy}
$$

其中：
- $c_1$：Critic 损失权重（通常 0.5）
- $c_2$：熵奖励权重（通常 0.01）
- $\epsilon$：clip 范围（通常 0.1~0.2）
- $\gamma$：折扣因子（通常 0.99~1.0）
- $\lambda$：GAE 参数（通常 0.95）
- $\beta$：KL 惩罚强度（通常 0.01~0.1）

---

## 九、各模型 Loss 一览表

| 模型 | Loss 公式 | 输入 | 目标 |
|------|----------|------|------|
| **Actor** | $-\frac{1}{n}\sum_t \min(\rho_t\hat{A}_t, \text{clip}(\rho_t)\hat{A}_t)$ | $x, y$ | 提高好 token 概率，压低差 token |
| **Critic** | $\frac{1}{n}\sum_t (V_\phi(x,y_{<t}) - \hat{V}_t^{target})^2$ | $x, y_{<t}$ | 准确预测价值 |
| **Reference** | 无 loss（冻结） | $x, y_{<t}$ | 提供 KL 基准 |
| **Reward** | 无 loss（冻结） | $x, y$ | 提供终端分数 |

---

## 十、协作关系图

```
prompt x
   │
   ▼
┌─────────┐   生成 y    ┌──────────┐
│  Actor  │───────────▶│  Reward  │──▶ r(x,y)
│ (训练)  │            │ (冻结)   │
└─────────┘            └──────────┘
   │  │
   │  │ logπ_θ(y_t)
   │  ▼
   │ ┌──────────┐
   │ │ Reference│──▶ logπ_ref(y_t) ──▶ KL惩罚
   │ │ (冻结)   │
   │ └──────────┘
   │
   │ 状态 s_t
   ▼
┌─────────┐
│  Critic │──▶ V(s_t) ──▶ 优势 Â_t
│ (训练)  │
└─────────┘
   │
   ▼
 PPO更新 Actor 和 Critic
```

---

## 十一、完整训练伪代码

```python
for iteration in range(N):
    # 1. 采集数据
    prompts = sample_batch()
    with torch.no_grad():
        responses, logprobs_old = actor.generate(prompts)
        logprobs_ref = reference.logprobs(prompts, responses)
        rewards = reward_model(prompts, responses)  # 终端标量
        values_old = critic(prompts, responses)     # 旧价值，不参与梯度

    # 2. 计算每步奖励
    kl = logprobs_old - logprobs_ref
    R = -beta * kl
    R[:, -1] += rewards  # 最后一步加偏好分

    # 3. GAE 算优势
    advantages = compute_gae(R, values_old, gamma, lambda)
    returns = advantages + values_old

    # 4. 多轮 PPO 更新
    for epoch in range(K):
        logprobs_new = actor.logprobs(prompts, responses)
        values_new = critic(prompts, responses)

        # 重要性比
        ratio = torch.exp(logprobs_new - logprobs_old)

        # Actor loss
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1-eps, 1+eps) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()

        # Critic loss
        critic_loss = ((values_new - returns) ** 2).mean()

        # 熵
        entropy = -(logprobs_new * torch.exp(logprobs_new)).sum(-1).mean()

        # 总 loss
        loss = actor_loss + c1 * critic_loss - c2 * entropy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```


## 十二、关键点总结

1. **Reward 模型**只给最后一步一个标量 $r$，注入偏好信号
2. **Reference 模型**提供每步 KL 惩罚，防止 Actor 跑偏
3. **Critic** 的 loss 是均方误差，target 是 $\hat{A}_t + V_{\phi_{old}}$，让价值估计越来越准
4. **Actor** 的 loss 是 clip 后的策略梯度，用 $\hat{A}_t$ 决定每个 token 该鼓励还是抑制
5. **总 loss** = Actor loss + $c_1$ × Critic loss − $c_2$ × Entropy

> **协作核心**：Reward 给终点分，Reference 给每步约束，Critic 把稀疏信号变成逐 token 优势，Actor 按优势更新。四者缺一不可。



## 十三、常见超参数速查

| 超参数 | 典型值 | 作用 |
|--------|--------|------|
| $\epsilon$ (clip) | 0.1~0.2 | 限制策略更新幅度 |
| $\gamma$ | 0.99~1.0 | 折扣因子 |
| $\lambda$ (GAE) | 0.95 | 偏差-方差权衡 |
| $\beta$ (KL) | 0.01~0.1 | 偏离 Reference 的惩罚强度 |
| $c_1$ (critic) | 0.5 | Critic 损失权重 |
| $c_2$ (entropy) | 0.01 | 熵奖励权重 |
| K (PPO epochs) | 4~10 | 每批数据重复更新次数 |
| batch size | 64~512 | 每次采集的 prompt 数 |



## 十四、总结

$$
\boxed{
L^{PPO} = L^{actor} + c_1 L^{critic} - c_2 L^{entropy}
}
$$

- **Actor**：用 clip 后的优势做策略梯度，稳定提升好动作概率
- **Critic**：回归价值目标 $\hat{A}_t + V_{\phi_{old}}$，为优势估计提供低方差基线
- **Entropy**：保持探索，防止过早收敛
- **Reward + Reference**：提供终端偏好信号和 KL 约束，定义每步奖励

四模型协作的最终产出，就是让 Actor 在“拿高分”和“不跑偏”之间找到平衡。