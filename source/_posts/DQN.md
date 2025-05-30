---
title: DQN
mathjax: true
toc: true
date: 2024-12-09 01:58:07
updated: 2024-12-09 01:58:07
categories:
- Reinforcement Learning
tags:
- Q-Learning
- Deep Q-Learning
---
最近我组有同学在探索用RL落地营销场景的可能性，借此机会学习下RL。

<!--more-->

## Q-Learning
Q-learning算法以表格的方式存储了每个状态下所有动作值的表格。表格中的每一个动作价值表示在状态下选择动作然后继续遵循某一策略预期能够得到的期望回报。Q值的更新公式如下：
$$
Q(S, A) \leftarrow Q(S, A)+\alpha\left[R+\gamma \max _a Q\left(S^{\prime}, A^{\prime}\right)-Q(S, A)\right]
$$
当 $Q(s,a)$ 不再显著变化时，算法收敛。其中：
- $S$：当前状态
- $A$：当前动作
- $R$：即时奖励
- $S'$：下一状态
- $\gamma$：折扣因子（$ 0 \leq \gamma \leq 1 $），控制未来奖励的重要性


然而，这种用表格存储动作价值的做法只在环境的状态和动作都是离散的，并且空间都比较小的情况下适用。当动作或状态数量巨大时，Q-Learning已捉襟见肘。

___
## Deep Q-Learning
Deep Q-Learning将Q-Table的更新问题变成一个函数拟合问题，相近的状态得到相近的输出动作。但DL与RL结合会存在如下四个问题：
1. DL需要大量带标签的样本进行监督学习，但RL只有reward返回值
   - 解决方案：使用reward构造标签
2. DL的样本独立，但RL前后state状态相关
   - 解决方案：使用Experience Replay
3. DL目标分布固定，但RL的分布一直变化。比如你玩一个游戏，一个关卡和下一个关卡的状态分布是不同的，所以训练好了前一个关卡，下一个关卡又要重新训练
   - 解决方案：使用Experience Replay
4. 使用非线性网络表示值函数时出现不稳定等问题，因为预估Q值和Target Q值都是由神经网络输出的
   - 解决方案：使用更新较慢的目标网络预估Target Q

整体的算法流程如下：

![DQN](https://github.com/TransformersWsz/picx-images-hosting/raw/master/dqn.175fokezb5.webp)

#### Experience Replay
使用经验回放有两点好处：
- 提高样本利用效率，也避免了灾难性遗忘
- 使样本满足独立假设。在MDP中交互采样得到的数据本身不满足独立假设，因为这一时刻的状态和上一时刻的状态有关。非独立同分布的数据对训练神经网络有很大的影响，会使神经网络拟合到最近训练的数据上。采用经验回放可以打破样本之间的相关性，让其满足独立假设。

#### Target Network
DQN算法最终更新的目标是让$Q_\theta(s,a)$逼近$r+\gamma \max _{a^{\prime}} Q_\theta\left(s^{\prime}, a^{\prime}\right)$，但由于TD误差目标本身就包含神经网络的输出，因此在更新网络参数的同时目标也在不断地改变，这非常容易造成神经网络训练的不稳定性。为了解决这一问题，DQN便使用了目标网络（target network）的思想：既然训练过程中Q网络的不断更新会导致目标不断发生改变，不如暂时先将TD目标中的Q网络固定住。为了实现这一思想，我们需要利用两套Q网络。

1. 原来的训练网络$Q_\theta(s,a)$，用于计算原来的损失函数中的项$\frac{1}{2}\left[Q_\theta(s, a)-\left(r+\gamma \max _{a^{\prime}} Q_{\theta^{-}}\left(s^{\prime}, a^{\prime}\right)\right)\right]^2$中的$Q_\theta(s,a)$，并且使用正常梯度下降方法来进行更新。
2. 目标网络$Q_{\theta^{-}}\left(s^{\prime}, a^{\prime}\right)$，用于计算原先损失函数中的$(r+\gamma \max _{a^{\prime}} Q_{\theta^{-}}\left(s^{\prime}, a^{\prime}\right))$项，其中$\theta^{-}$表示目标网络中的参数。如果两套网络的参数随时保持一致，则仍为原先不够稳定的算法。为了让更新目标更稳定，目标网络并不会每一步都更新。具体而言，目标网络使用训练网络的一套较旧的参数，训练网络$Q_\theta(s,a)$在训练中的每一步都会更新，而目标网络的参数每隔$C$步才会与训练网络同步一次，即$\theta^{-} \leftarrow \theta$。这样做使得目标网络相对于训练网络更加稳定。

#### 代码实践
具体的交互环境搭建和DQN代码见：[5_Deep_Q_Network](https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/5_Deep_Q_Network/RL_brain.py)
___

## 参考
- [实战深度强化学习DQN-理论和实践](https://cloud.tencent.com/developer/article/1092239)
- [Simple Reinforcement learning tutorials](https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow)
- [什么是DQN](https://mofanpy.com/tutorials/machine-learning/reinforcement-learning/intro-DQN)
- [The Deep Q-Learning Algorithm](https://huggingface.co/learn/deep-rl-course/unit3/deep-q-algorithm)
- [Human-level control through deep reinforcement learning](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
- [第7章 DQN 算法](https://hrl.boyuai.com/chapter/2/dqn%E7%AE%97%E6%B3%95)