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
Q-learning算法以表格的方式存储了每个状态下所有动作值的表格。


然而，这种用表格存储动作价值的做法只在环境的状态和动作都是离散的，并且空间都比较小的情况下适用，我们之前进行代码实战的几个环境都是如此（如悬崖漫步）。当状态或者动作数量非常大的时候，这种做法就不适用了。例如，当状态是一张 RGB 图像时，假设图像大小是，此时一共有种状态，在计算机中存储这个数量级的值表格是不现实的。更甚者，当状态或者动作连续的时候，就有无限个状态动作对，我们更加无法使用这种表格形式来记录各个状态动作对的值。

___

## 参考
- [实战深度强化学习DQN-理论和实践](https://cloud.tencent.com/developer/article/1092239)
- [Simple Reinforcement learning tutorials](https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow)
- [什么是DQN](https://mofanpy.com/tutorials/machine-learning/reinforcement-learning/intro-DQN)
- [The Deep Q-Learning Algorithm](https://huggingface.co/learn/deep-rl-course/unit3/deep-q-algorithm)
- [Human-level control through deep reinforcement learning](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
- [第7章 DQN 算法](https://hrl.boyuai.com/chapter/2/dqn%E7%AE%97%E6%B3%95)