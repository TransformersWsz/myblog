---
title: DPO讲解
mathjax: true
toc: true
date: 2026-09-29 01:24:01
updated: 2026-09-29 01:24:01
categories:
- NLP
tags:
- LLM
- DPO
- RM
---
PPO算法的pipeline冗长，涉及模型多，资源消耗大，且训练极其不稳定。DPO是斯坦福团队基于PPO推导出的优化算法，去掉了RW训练和RL环节，只需要加载一个推理模型和一个训练模型，直接在偏好数据上进行训练即可：

![DPO](https://github.com/TransformersWsz/picx-images-hosting/raw/master/image.suiljdpc9dc.png)

<!--more-->

损失函数如下(顿时清爽简洁了不少)：
$$
\mathcal{L}_{\mathrm{DPO}}\left(\pi_\theta ; \pi_{\mathrm{ref}}\right)=-\mathbb{E}_{\left(x, y_w, y_l\right) \sim \mathcal{D}}\left[\log \sigma\left(\beta \log \frac{\pi_\theta\left(y_w \mid x\right)}{\pi_{\mathrm{ref}}\left(y_w \mid x\right)}-\beta \log \frac{\pi_\theta\left(y_l \mid x\right)}{\pi_{\mathrm{ref}}\left(y_l \mid x\right)}\right)\right]
$$

可以拆成两部分来看：
- **最大化好答案的奖励值**：$\text{max} \log \frac{\pi_\theta\left(y_w \mid x\right)}{\pi_{\mathrm{ref}}\left(y_w \mid x\right)}$
- **最小化好答案的奖励值**：$\text{min} \log \frac{\pi_\theta\left(y_l \mid x\right)}{\pi_{\mathrm{ref}}\left(y_l \mid x\right)}$

两部分组合就是DPO的公式了。

DPO在理解难度、实现难度和资源占用都非常友好，想看具体的公式推导见：

[[论文笔记]DPO：Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://zhuanlan.zhihu.com/p/653975451)

## DPO的缺陷
DPO 为什么要用同分布数据？
在搞 DPO 时，很多人会踩一个坑：拿 GPT-5 的数据去 DPO 一个 7B 的小模型，结果发现效果并不好。为什么？这里就涉及到一个核心概念：同分布（On-Policy / Near-Policy）。

通俗点说，DPO 的数据最好是自己改自己的错题集。

- **别人的错题（分布差异大）**：
如果你拿 Gemini3 Pro 生成的“坏答案”给 7B 小模型看，小模型可能会一脸懵逼：“这句坏答案写得文采飞扬、逻辑复杂，有没可能虽然它是错的，但我本来就写不出来啊？”
既然模型本来生成这句话的概率就极低，那你再去压低它的概率，对模型来说就是无效补课，梯度几乎为 0。
- **自己的错题（同分布）**：
只有拿模型自己（或者能力相近的模型）生成的坏答案，它才会恍然大悟：“哎呀，这确实是我经常忍不住想说的那句蠢话！原来这是错的，我改！”

所以，DPO 想要效果好，最好是用当前模型去采样数据，然后让人类（或强模型）去标注好坏。这种量身定做的错题集，才是提分的关键。

___

## 参考
- [Direct Preference Optimization:
Your Language Model is Secretly a Reward Model](https://arxiv.org/pdf/2305.18290.pdf)
- [DPO: Direct Preference Optimization 论文解读及代码实践](https://zhuanlan.zhihu.com/p/642569664)