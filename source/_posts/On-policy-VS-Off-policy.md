---
title: On-policy VS Off-policy
mathjax: true
toc: true
date: 2026-03-01 18:11:36
updated: 2026-03-01 18:11:36
categories:
- Reinforcement Learning
tags:
- On-policy
- Off-policy
---

此前对于rl的这两个概念一直很模糊，在此整理一下。

<!--more-->

首先介绍下行为策略和目标策略：
- 行为策略(Behavior Policy)：用来收集数据的策略，也就是实际与环境交互、生成样本的策略。
- 目标策略(Target Policy)：我们要学习和优化的策略，也就是最终想要得到的策略。

> 如果行为策略与目标策略一致，则是on-policy，否则为off-policy。

下面将以两个经典算法为例介绍on-policy、off-policy区别。

## SARSA(On-policy)

算法流程如下：

![sarsa](https://raw.githubusercontent.com/TransformersWsz/picx-images-hosting/master/image.9rjxxkigy3.webp)


python伪代码如下：

```python
# SARSA 算法
def sarsa():
    # 初始化 Q 表
    Q = {}
    
    for episode in range(N):
        state = env.reset()
        
        # ⚠️ 关键点1：用 ε-greedy 选第一个动作
        action = epsilon_greedy(Q, state)  # 这就是当前的行为策略
        
        while not done:
            # 执行动作，得到下一状态和奖励
            next_state, reward, done = env.step(action)
            
            # ⚠️ 关键点2：还是用 ε-greedy 选下一个动作
            next_action = epsilon_greedy(Q, next_state)  # 还是当前的行为策略
            
            # ⚠️ 关键点3：更新时用的是下一个动作的 Q 值
            # SARSA 更新公式
            Q[state][action] += lr * (reward + gamma * Q[next_state][next_action] - Q[state][action])
            
            state, action = next_state, next_action
```


| 策略类型 | 是什么 | 在代码中的位置 |
|------|------|------|
| **行为策略** | ε-greedy | `epsilon_greedy(Q, state)` |
| **目标策略** | ε-greedy | 同上，就是同一个策略 |

收集数据时用的是 ε-greedy，更新时用的还是 ε-greedy 选的 next_action，两者策略一致，是on-policy。

## DQN(Off-policy)

算法流程如下：



![dqn](https://raw.githubusercontent.com/TransformersWsz/picx-images-hosting/master/image.491thfbbvt.webp)

python伪代码如下：
```python
# DQN 算法
def dqn():
    Q = Network()  # Q网络
    replay_buffer = []  # 经验池
    
    for episode in range(N):
        state = env.reset()
        
        while not done:
            # ⚠️ 关键点1：行为策略是 ε-greedy
            action = epsilon_greedy(Q, state)  # 这就是当前的行为策略
            
            # 执行动作，存经验
            next_state, reward, done = env.step(action)
            replay_buffer.append((state, action, reward, next_state, done))
            
            # ⚠️ 关键点2：从经验池采样（可能是旧策略的数据）
            batch = random.sample(replay_buffer, 32)
            
            for s, a, r, s_next, d in batch:
                # ⚠️ 关键点3：目标策略是 greedy！
                target = r + gamma * max(Q_target(s_next))  # 用 max，不是 ε-greedy！
                
                # 更新 Q 网络
                loss = MSE(Q(s)[a], target)
                loss.backward()
            
            state = next_state
```

| 策略类型 | 是什么 | 在代码中的位置 |
|------|------|------|
| **行为策略** | ε-greedy | 收集数据时的策略 |
| **目标策略** | **greedy** | 更新时的 `max(Q_target(s_next))` |

收集数据用的ε-greedy，但更新永远选最好的动作greedy，两者策略不一致，是off-policy。

___

## 参考
- [强化学习基础3：一文彻底讲清On-policy与Off-policy](https://zhuanlan.zhihu.com/p/26603719923)