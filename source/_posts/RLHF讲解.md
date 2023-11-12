---
title: RLHF讲解
mathjax: true
toc: true
date: 2023-11-13 02:15:29
categories:
- NLP
tags:
- LLM
- PPO
- RM
- Actor-Critic
---
RLHF包含了两个至关重要的步骤：
1. 训练Reward Model
2. 用Reward Model和SFT Model构造Reward Function，基于PPO算法来训练LLM
   1. frozen RM
   2. frozen SFT Model
   3. Actor $\pi_{\Phi}^{R L}$ initialized from SFT Model
   4. Critic $V_\eta$ initialized from RM


<!--more-->

![rlhf](https://github.com/TransformersWsz/picx-images-hosting/raw/master/image.6qiivvmcc5c0.png)

___

## 参考
[RLHF理论篇](https://zhuanlan.zhihu.com/p/657490625)