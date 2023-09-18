---
title: InstructGPT
mathjax: true
toc: true
date: 2023-07-09 02:46:42
categories:
- Machine Learning
tags:
- LLM
- Reinforcement Learning
- GPT
- PPO
---

ChatGPT背后的技术原理：

<!--more-->

![InstructGPT](https://github.com/TransformersWsz/picx-images-hosting/raw/master/image.2mzdjib60zy0.webp)

- 第二步中已经完成了奖励模型的训练，在第三步中奖励模型用PPO来训练第一步中微调好的GPT3，使其能够生成符合指令的文本