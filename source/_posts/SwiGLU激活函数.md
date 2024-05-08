---
title: SwiGLU激活函数
mathjax: true
toc: true
date: 2024-05-09 01:00:36
updated: 2024-05-09 01:00:36
categories:
- NLP
tags:
- LLM
- 激活函数
- GELU
- Swish
- GLU
---
SwiGLU激活函数已经成为LLM的标配了。它是GLU的变体，公式如下：
$$
\operatorname{SwiGLU}(x, W, V, b, c, \beta)=\operatorname{Swish}_\beta(x W+b) \otimes(x V+c)
$$

<!--more-->

## Swish
$$
\operatorname{Swish_\beta}(x)=x \otimes \sigma(\beta x)
$$
在nlp和cv任务上，Swish性能都和GELU接近，稍微略高点。但Swish公式更简洁优雅。

GELU早期被BERT、RoBERTa、ALBERT采用。

## GLU
$$
\operatorname{GLU}(x, W, V, b, c)=\sigma(x W+b) \otimes(x V+c)
$$
单纯从公式看，GLU是一个神经网络层。左右两个线性变换层，左边再接一个门控机制来控制信息流通多少。

## SwiGLU
将Swish作为左侧激活函数就得到了SwiGLU。代码如下：
```python
F.silu(self.w1(x)) * self.w2(x)
```

在 [GLU Variants Improve Transformer](https://arxiv.org/pdf/2002.05202) 论文中，作者比较了各种GLU变体的激活函数，SwiGLU在各项任务上表现出众。但作者并未给出解释原因，只能说后验是这样，那就选它呗，所以成了LLM的标配。

## 各激活函数示意图
![act](https://raw.githubusercontent.com/TransformersWsz/picx-images-hosting/master/image.8kzwfgcqya.png)
___

## 参考
- [GLU Variants Improve Transformer](https://arxiv.org/pdf/2002.05202)
- [SWISH: A SELF-GATED ACTIVATION FUNCTION](https://arxiv.org/pdf/1710.05941v1)
- [超越ReLU却鲜为人知，3年后被挖掘：BERT、GPT-2等都在用的激活函数](https://www.jiqizhixin.com/articles/2019-12-30-4)
- [大模型基础｜激活函数｜从ReLU 到SwiGLU](https://zhuanlan.zhihu.com/p/650237644)