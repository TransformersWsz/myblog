---
title: PEFT-LISA
mathjax: true
toc: true
date: 2024-04-03 01:16:50
updated: 2024-04-03 01:16:50
categories:
- NLP
tags:
- LLM
- PEFT
---
LISA是LoRA的简化版，但其抓住了LoRA微调的核心，即LoRA侧重更新LLM的底层embedding和顶层head。

<!--more-->

![phe](https://github.com/TransformersWsz/picx-images-hosting/raw/master/image.4uaperxni9.png)

根据上述现象，LISA提出两点改进：
- 始终更新LLM的底层embedding和顶层head
- 随机更新中间层的hidden state

![phe](https://github.com/TransformersWsz/picx-images-hosting/raw/master/image.6f0ge908x1.webp)

## 实验结果

#### 显存占用
![gpu](https://github.com/TransformersWsz/picx-images-hosting/raw/master/image.4cknq74f33.webp)

毕竟模型参数大头还是在底层embedding，所以显存占用并没有减少太多。

#### 训练时间
![time](https://github.com/TransformersWsz/picx-images-hosting/raw/master/image.8ad16vg8o3.webp)

#### 下游任务微调
![time](https://github.com/TransformersWsz/picx-images-hosting/raw/master/image.101xvtqs27.webp)

在MT-BENCH上，LISA超过了LoRA，甚至全量参数微调。

___

## 参考
- [比LoRA还快50%的微调方法来了！一张3090性能超越全参调优，UIUC联合LMFlow团队提出LISA](https://mp.weixin.qq.com/s/7s8NNGYlq4JWeln0TkOKmQ)
- [LMFlow](https://github.com/OptimalScale/LMFlow)
