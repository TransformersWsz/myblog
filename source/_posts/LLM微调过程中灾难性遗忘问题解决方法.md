---
title: LLM微调过程中灾难性遗忘问题解决方法
mathjax: true
toc: true
date: 2023-11-06 02:35:56
categories:
tags:
---

灾难性遗忘是LLM微调过程中最常见的问题，下面是一些解决办法：

<!--more-->

1. 将重要的权重冻结：像Lora就是采用的这种方案，只学习部分网络权重。但这里Lora的配置其实是要注意一下，如果你是用Lora做预训练，lora训练模块可以配上 q_proj,v_proj,k_proj,o_proj  如果是微调则只需要训练q_proj,v_proj。lora_rank的设置也有讲究，初始设lora_ran为8，训练存在遗忘时，可以将 lora_rank改为64（原因是与原模型数据领域相差较大的话，需要更大的秩，原论文有说明）
2. 复习：跟人一样，在预训练或微调时，回看之前训练的数据。还可以专门把特征图存起来，量化以后放在一个类似于记忆库的地方，之后在新任务上训练的时候从这个记忆库里重构出记忆和新数据一起训练。感兴趣可以看这篇论文：[REMIND Your Neural Network to Prevent
Catastrophic Forgetting](https://arxiv.org/pdf/1910.02509.pdf)
3. MoE：稀疏门控制的专家混合层，最近爆出GPT4是由 8个220B的模型组合。但个人体验，阉割版的GPT4变得智障了很多。
4. 数据蒸馏：损失函数由teacher-student的KL loss和groud truth label构成：https://github.com/beyondguo/LLM-Tuning/discussions/24

___

## 参考
- [大语言模型Fine-tuning踩坑经验之谈](https://mp.weixin.qq.com/s/Aa8jYs4xgcI4clwie-wO1g)