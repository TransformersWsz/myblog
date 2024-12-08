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
Q(S, A) \leftarrow Q(S, A)+\alpha\left[R+\gamma \max _a Q\left(S^{\prime}, a\right)-Q(S, A)\right]
$$

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
4. 使用非线性网络表示值函数时出现不稳定等问题

___

## 参考
- [实战深度强化学习DQN-理论和实践](https://cloud.tencent.com/developer/article/1092239)
- [Simple Reinforcement learning tutorials](https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow)
- [什么是DQN](https://mofanpy.com/tutorials/machine-learning/reinforcement-learning/intro-DQN)
- [The Deep Q-Learning Algorithm](https://huggingface.co/learn/deep-rl-course/unit3/deep-q-algorithm)
- [Human-level control through deep reinforcement learning](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
- [第7章 DQN 算法](https://hrl.boyuai.com/chapter/2/dqn%E7%AE%97%E6%B3%95)