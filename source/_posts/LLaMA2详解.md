---
title: LLaMA2详解
mathjax: true
toc: true
date: 2024-05-23 02:19:35
updated: 2024-05-23 02:19:35
categories:
- NLP
tags:
- LLM
- LLaMA
---

## padding_side
llama系列训练和推理都是right padding：
- 训练：其实只要设置padding mask，那么left/right pad是没有区别的。然而实际上huggingface中某些tokenizer在training的时候必须设成right padding，因为有些tokenizer使用的是absolute position id，导致非padding token的position id不是从0开始的。
- 推理：由于llama是decoder-only结构，每个token的生成依赖于上一个token。而上一个token如果是无实际意义的padding token，那么就打断了句子语义的连续性，导致生成文本质量较差。因此left padding比较make sense，但在llama推理的源码中，batch的时候采用[right padding](https://github.com/meta-llama/llama3/blob/14aab0428d3ec3a9596f1dea06d9c564f9c0e35f/llama/generation.py#L155)。生成token的时候，[从batch中长度最短的序列开始生成，其它较长序列的相同位置保持不变](https://github.com/meta-llama/llama3/blob/14aab0428d3ec3a9596f1dea06d9c564f9c0e35f/llama/generation.py#L184)，直到该序列开始生成。猜测llama这样做的原因是保持跟训练一致，但不如left padding优雅。

## 参考
- [大部分的大模型(LLM)采用左填充(left-padding)的原因](https://zhuanlan.zhihu.com/p/646852375)