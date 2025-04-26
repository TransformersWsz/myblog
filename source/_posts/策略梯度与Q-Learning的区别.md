---
title: 策略梯度与Q-Learning的区别
mathjax: true
toc: true
date: 2025-04-27 02:33:34
updated: 2025-04-27 02:33:34
categories:
- Reinforcement Learning
tags:
- Q-Learning
- Policy Gradient
---
PG和Q-Learning都是RL的两大主流算法，记录下两者差异。

<!--more-->

## 策略梯度（Policy Gradient）简介
策略梯度（Policy Gradient, PG）是强化学习中的一类直接优化策略的方法，通过梯度上升（Gradient Ascent）更新策略参数，以最大化期望回报。与Q-Learning等基于值函数的方法不同，PG直接对策略 $\pi_\theta(a|s)$（参数为$\theta$）进行优化，适用于连续动作空间或随机策略的场景。

---

### 核心思想
1. 策略参数化：  
   用神经网络或其他函数近似策略 $\pi_\theta(a|s)$，输入状态s，输出动作a的概率分布（或连续动作的均值/方差）。
2. 目标函数：  
   最大化期望回报 $J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} [R(\tau)]$，其中$\tau$是轨迹（状态-动作序列），$R(\tau)$是轨迹的总奖励。
3. 梯度上升：  
   计算目标函数对策略参数\theta的梯度 $\nabla_\theta J(\theta)$，并沿梯度方向更新参数：
$$
\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)
$$


### 策略梯度定理
梯度 $\nabla_\theta J(\theta)$ 的数学形式为：
$$
\nabla_\theta J(\theta)=\mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^T \nabla_\theta \log \pi_\theta\left(a_t \mid s_t\right) \cdot Q^{\pi_\theta}\left(s_t, a_t\right)\right]
$$
- $\log \pi_\theta(a_t|s_t)$：策略的对数概率。  
- $Q^{\pi_\theta}(s_t, a_t)$：状态-动作值函数（即从$s_t$执行$a_t$后的累计奖励期望）。

### 经典算法：REINFORCE
REINFORCE 是最简单的策略梯度算法，通过蒙特卡洛采样估计梯度：
1. 用当前策略 $\pi_\theta$ 生成一条轨迹 $\tau = (s_0, a_0, r_0, \dots, s_T)$。
2. 计算轨迹的累计奖励 $R(\tau) = \sum_{t=0}^T \gamma^t r_t$（$\gamma$为折扣因子）。
3. 更新策略参数：
$$
\theta \leftarrow \theta+\alpha \gamma^t R(\tau) \nabla_\theta \log \pi_\theta\left(a_t \mid s_t\right)
$$


### 举例说明
CartPole（平衡杆问题）
- 目标：控制小车左右移动，使杆子保持直立不倒。
- 状态$s$：小车位置、速度、杆子角度、角速度。
- 动作$a$：离散（左移/右移）或连续（施加的力）。
- 奖励$r$：每步杆子未倒下时+1，倒下后终止。

PG实现步骤（以REINFORCE为例）
1. 初始化策略网络：  
   输入状态s，输出动作概率（如左移概率0.7，右移0.3）。
2. 采样轨迹：  
   根据当前策略运行游戏，得到轨迹 $\tau = (s_0, a_0, r_0, \dots, s_T)$。
3. 计算梯度：  
   对每一步t，计算 $\nabla_\theta \log \pi_\theta(a_t|s_t)$，乘以累计奖励 $R(\tau)$。
4. 更新策略：  
   沿梯度方向调整 $\theta$，使高奖励动作的概率增加。

经过多次迭代，策略会学会在杆子右倾时选择左移动作（反之亦然），最终保持平衡。

**策略梯度的优缺点**
• 优点：  

  • 直接优化策略，适用于连续动作空间（如机器人控制）。  

  • 可学习随机策略（如石头剪刀布游戏）。  

• 缺点：  

  • 高方差（需大量采样或改进算法如PPO、A3C）。  

  • 可能收敛到局部最优。


---

**改进算法**
• Actor-Critic：引入值函数（Critic）减少方差。  

• PPO（近端策略优化）：通过约束策略更新步长提升稳定性。  

• A3C：异步并行采样加速训练。


策略梯度是深度强化学习（如AlphaGo）的基础方法之一，结合神经网络后可解决复杂任务。