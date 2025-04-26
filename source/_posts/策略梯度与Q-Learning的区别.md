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
1. 策略参数化：用神经网络或其他函数近似策略 $\pi_\theta(a|s)$，输入状态s，输出动作a的概率分布（或连续动作的均值/方差）。
2. 目标函数：最大化期望回报 $J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} [R(\tau)]$，其中$\tau$是轨迹（状态-动作序列），$R(\tau)$是轨迹的总奖励。
3. 梯度上升：计算目标函数对策略参数\theta的梯度 $\nabla_\theta J(\theta)$，并沿梯度方向更新参数：
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


### 举例1（离散型动作）
**CartPole（平衡杆问题）**
- 目标：控制小车左右移动，使杆子保持直立不倒。
- 状态$s$：小车位置、速度、杆子角度、角速度。
- 动作$a$：离散（左移/右移）或连续（施加的力）。
- 奖励$r$：每步杆子未倒下时+1，倒下后终止。

**PG实现步骤（以REINFORCE为例）**
1. 初始化策略网络：输入状态s，输出动作概率（如左移概率0.7，右移0.3）。
2. 采样轨迹：根据当前策略运行游戏，得到轨迹 $\tau = (s_0, a_0, r_0, \dots, s_T)$。
3. 计算梯度：对每一步t，计算 $\nabla_\theta \log \pi_\theta(a_t|s_t)$，乘以累计奖励 $R(\tau)$。
4. 更新策略：沿梯度方向调整 $\theta$，使高奖励动作的概率增加。

经过多次迭代，策略会学会在杆子右倾时选择左移动作（反之亦然），最终保持平衡。


### 举例2（连续型动作）
当动作空间是连续的（例如机器人控制、自动驾驶中的转向角度等），策略梯度方法可以通过以下方式输出连续动作：

1. 参数学习：策略网络 $\pi_\theta(a|s)$ 不再输出离散动作的概率分布，而是输出连续动作的概率分布参数，通常选择高斯分布（正态分布），其参数为：
   - 均值 $\mu$：表示动作的中心值（如转向角度为0.5弧度）。
   - 标准差 $\sigma$（或对数标准差 $\log \sigma$）：表示动作的探索范围（$\sigma=0.1$表示小幅随机扰动）。

代码示例：
```python
import torch
import torch.nn as nn

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.mu_head = nn.Linear(64, action_dim)    # 输出均值
        self.log_std_head = nn.Linear(64, action_dim)  # 输出对数标准差

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        mu = torch.tanh(self.mu_head(x))  # 均值限制在[-1,1]（假设动作范围）
        log_std = self.log_std_head(x)    # 对数标准差
        return mu, log_std
```

2. 动作采样：在状态$s$下，策略网络输出均值和标准差后，通过重参数化技巧采样动作：
$$
a \sim \mathcal{N}(\mu_\theta(s), \sigma_\theta(s))
$$
代码示例：
```python
def select_action(state):
    mu, log_std = policy_network(state)
    std = torch.exp(log_std)  # 保证标准差为正
    normal_dist = torch.distributions.Normal(mu, std)
    action = normal_dist.rsample()  # 可导的采样
    return action.clamp(-1.0, 1.0)  # 限制动作范围
```
3. 策略梯度的计算：连续动作空间的策略梯度公式与离散情况类似，但需使用连续概率密度函数（PDF）的对数：
$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot Q^{\pi_\theta}(s_t, a_t) \right]
$$
其中 $\log \pi_\theta(a_t|s_t)$ 是高斯分布的对数概率密度：
$$
\log \pi_\theta(a_t|s_t) = -\frac{(a_t - \mu_\theta(s_t))^2}{2 \sigma_\theta(s_t)^2} - \log \sigma_\theta(s_t) - \text{常数}
$$
代码示例：
```python
def compute_loss(states, actions, rewards):
    mu, log_std = policy_network(states)
    std = torch.exp(log_std)
    normal_dist = torch.distributions.Normal(mu, std)
    log_probs = normal_dist.log_prob(actions)  # 计算对数概率
    loss = -log_probs * rewards  # 梯度上升转化为梯度下降
    return loss.mean()
```

**具体例子：连续动作的平衡杆问题**

假设我们需要控制小车的连续推力（范围$[-1, 1]$）：
1. 状态$s$：杆子角度、角速度、小车位置、速度。
2. 动作$a$：推力值（如0.3表示向右的力，-0.8表示向左的力）。
3. 策略网络：输出均值$\mu \in [-1,1]$和标准差$\sigma$（探索噪声）。
4. 训练过程：
   - 采样动作 $a \sim \mathcal{N}(\mu, \sigma)$。
   - 执行动作后获得奖励（如杆子保持直立的时间）。
   - 用策略梯度更新网络，使高奖励动作的$\mu$向$a$靠近，同时自适应调整$\sigma$。


### 总结
- 离散动作：策略网络输出离散动作的概率分布（如Softmax）。
- 连续动作：策略网络输出高斯分布的参数（$\mu, \sigma$），通过采样得到连续值。  
- 核心公式：$\nabla_\theta J(\theta) = \mathbb{E}[\nabla_\theta \log \pi_\theta(a|s) \cdot Q(s,a)]$，其中$\log \pi_\theta(a|s)$需按连续分布计算。

___

## PG与Q-Learning差异

以下是两者的核心对比：

![diff](https://github.com/TransformersWsz/picx-images-hosting/raw/master/image.64e2128tb1.webp)


### 举例说明
**场景：CartPole（平衡杆问题）**
- 状态$s$：杆子角度、角速度、小车位置、速度。
- 动作$a$：离散（左/右移动）或连续（施加的力）。
- 奖励$r$：每步杆子未倒下时+1，倒下后终止。


#### Q-Learning 的实现（离散动作）
1. Q表或Q网络：维护 $Q(s,a)$，输入状态$s$，输出每个动作的Q值（如左/右）。  
2. 动作选择：
   - 训练时：ε-greedy（以概率ε随机探索，否则选$\arg\max_a Q(s,a)$）。  
   - 测试时：纯贪婪策略 $\arg\max_a Q(s,a)$。  

3. 更新规则（TD学习）：  
   $$
   Q(s,a) \leftarrow Q(s,a) + \alpha \left[ r + \gamma \max_{a'} Q(s',a') - Q(s,a) \right]
   $$
   其中 $s'$ 是下一状态，$\gamma$ 是折扣因子。

代码示例（DQN）：
```python
import numpy as np

# Q网络预测Q值
q_values = q_network(state)
# ε-greedy选择动作
if np.random.rand() < epsilon:
    action = env.action_space.sample()  # 随机探索
else:
    action = np.argmax(q_values)  # 贪婪动作
# 更新Q网络（最小化TD误差）
loss = (target_q - q_values[action]) ** 2
```

特点：  
- 只能处理离散动作（如左/右）。  
- 策略隐含在Q值中（$\arg\max Q$ 是确定性策略）。  


#### 策略梯度的实现（连续动作）
1. 策略网络：输出动作分布参数（如高斯分布的均值$\mu$和标准差$\sigma$）。  
2. 动作采样：
   - 从分布中采样连续动作 $a \sim \mathcal{N}(\mu, \sigma)$。  
   - 例如：输出推力值 $a \in [-1,1]$。  
3. 更新规则（REINFORCE）：  
   $$
   \theta \leftarrow \theta + \alpha \nabla_\theta \log \pi_\theta(a|s) \cdot R(\tau)
   $$
   其中 $R(\tau)$ 是轨迹的总奖励。

代码示例（PG）：
```python
import torch

# 策略网络输出均值和标准差
mu, log_std = policy_network(state)
std = torch.exp(log_std)
normal_dist = torch.distributions.Normal(mu, std)
action = normal_dist.rsample()  # 可导采样
# 计算对数概率
log_prob = normal_dist.log_prob(action)
# 更新策略网络（最大化期望奖励）
loss = -log_prob * discounted_reward
```

特点：  
- 直接输出连续动作（如推力值0.73）。  
- 策略本身是随机的（通过$\sigma$控制探索）。  


### 关键区别总结
1. 策略 vs Q值：  
   - PG显式学习策略 $\pi_\theta(a|s)$，适合复杂动作空间（如机器人控制）。  
   - Q-Learning隐式策略依赖Q值最大化，难以处理连续动作（需DDPG等扩展）。  
2. 探索方式：  
   - PG通过策略的随机性自然探索（如高斯噪声）。  
   - Q-Learning需人工设计探索（如ε-greedy）。  
3. 适用场景：  
   - PG：连续控制（如机械臂抓取）、需要随机策略的任务（如博弈）。  
   - Q-Learning：离散动作空间（如游戏AI、广告推荐）。  