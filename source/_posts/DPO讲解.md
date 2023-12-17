---
title: DPO讲解
mathjax: true
toc: true
date: 2023-12-18 01:24:01
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

损失函数如下：
$$
\mathcal{L}_{\mathrm{DPO}}\left(\pi_\theta ; \pi_{\mathrm{ref}}\right)=-\mathbb{E}_{\left(x, y_w, y_l\right) \sim \mathcal{D}}\left[\log \sigma\left(\beta \log \frac{\pi_\theta\left(y_w \mid x\right)}{\pi_{\mathrm{ref}}\left(y_w \mid x\right)}-\beta \log \frac{\pi_\theta\left(y_l \mid x\right)}{\pi_{\mathrm{ref}}\left(y_l \mid x\right)}\right)\right]
$$

DPO在理解难度、实现难度和资源占用都非常友好，想看具体的公式推导见：

[[论文笔记]DPO：Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://zhuanlan.zhihu.com/p/653975451)

___

## 参考
- [Direct Preference Optimization:
Your Language Model is Secretly a Reward Model](https://arxiv.org/pdf/2305.18290.pdf)
- [DPO: Direct Preference Optimization 论文解读及代码实践](https://zhuanlan.zhihu.com/p/642569664)