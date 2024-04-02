---
title: PEFT
mathjax: true
toc: true
date: 2023-07-18 23:57:18
updated: 2023-07-18 23:57:18
categories:
- NLP
tags:
- LLM
- PEFT
---

下面是一些参数高效的微调大模型方法：

<!--more-->

## Adapter

#### 模型总览
![Adapter](https://github.com/TransformersWsz/picx-images-hosting/raw/master/image.1ocq1j9sds1s.webp)

Adapter作为一个插件加入到大模型内，微调下游任务时，固定大模型参数，只训练Adapter参数。

## LoRA
LoRA名为大语言模型的低阶适应，最初设计用于微调LLM，但却在文生图领域大放异彩，并逐渐被人数知。其思想跟ResNet非常相似，通过在大模型旁侧添加一路分支，冻结大模型参数，学习分支参数（也即残差），达到微调效果。

#### 模型总览
![lora](https://github.com/TransformersWsz/picx-images-hosting/raw/master/image.2nti7ywk7se0.webp)

![formula](https://github.com/TransformersWsz/picx-images-hosting/raw/master/image.6rua2cocjfs0.webp)

如果 $\Delta W$ 跟 $W_0$ 一样，也是 $\mathbb{R}^{d \times d}$，那么残差学习同样需要训练大量的参数，并没有达到参数高效的目标。而在我们学习中，常用的减少矩阵参数大小方法就是矩阵分解，因此作者对输入先降采样，再上采样，实现输入与输出维度一致。

## Prefix-Tuning
该方法主要用来做NLG任务（Table-to-text Generation、 Summarization），在输入token之前构造一段任务相关的virtual tokens作为Prefix，然后训练的时候只更新Prefix部分的参数，而大模型参数冻结。

#### 模型总览
![prefix tuning](https://github.com/TransformersWsz/picx-images-hosting/raw/master/image.1lj0d2s95wsg.webp)

Prefix tokens初始化如下：

![init](https://github.com/TransformersWsz/picx-images-hosting/raw/master/image.lq7qaw0w668.webp)

需要注意的是，在低资源场景下，用任务相关的单词来初始化prefix tokens，效果更好：

![words](https://github.com/TransformersWsz/picx-images-hosting/raw/master/image.36f27jncegw0.webp)

## Prompt-tuning
Prompt-Tunning算是prefix-Tunning的简化版本，面向NLU任务，进行了更全面的效果对比，并且在大模型上成功打平了LM微调的效果。

#### 模型总览
![prompt tuning](https://github.com/TransformersWsz/picx-images-hosting/raw/master/image.3w6seu3s3tm0.webp)

- 初始化：Prompt-tuning在输入层前置多个可训练的tokens，固定住大模型参数。实验结果表明用类标签来初始化prompts效果最好。
- prompt ensembling：针对同一个任务，构造多个不同的prompts，就相当于训练了多个模型。
  
![ensemble](https://github.com/TransformersWsz/picx-images-hosting/raw/master/image.5md9psbb9i40.webp)

#### Prompt-tuning 与 Prefix-Tuning 不同
- 两者的基座模型不同，一个是T5，一个是BART和GPT2
- 前者关注NLU，后者关注NLG
- 前者参数更少，只需微调embeding层；后者需要微调所有层embedding，以及需要在输入层之后接一个MLP来稳定训练

## P-tuning V1 & P-tuning V2
P-tuning主要用GPT来做NLU任务，达到甚至超过BERT同等水平。

#### 模型总览
![v1](https://github.com/TransformersWsz/picx-images-hosting/raw/master/image.6krujsjxqj40.webp)

v1做了如下两点优化：
- 考虑到预训练模型本身的embedding就比较离散了（随机初始化+梯度传回来小，最后只是小范围优化），同时prompt本身也是互相关联的，所以作者先用LSTM对prompt进行编码。
- 在prompt模板中，加入一些anchor tokens效果会更好。

v2主要是在大模型的每一层加入可训练prompts：

![v2](https://github.com/TransformersWsz/picx-images-hosting/raw/master/image.77uorheyphc0.webp)
___

## 参考
- [解密Prompt系列3. 冻结LM微调Prompt: Prefix-Tuning & Prompt-Tuning & P-Tuning](https://www.cnblogs.com/gogoSandy/p/17202169.html)
- [Prompt范式第二阶段｜Prefix-tuning、P-tuning、Prompt-tuning](https://zhuanlan.zhihu.com/p/400790006)
- [大模型参数高效微调技术原理综述（四）-Adapter Tuning及其变体](https://juejin.cn/post/7242677017057755191)