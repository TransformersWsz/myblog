---
title: GPT的实现细节
mathjax: true
toc: true
date: 2024-03-18 01:32:12
updated: 2024-03-18 01:32:12
categories:
- NLP
tags:
- LLM
- GPT
---

关于GPT的代码细节，这里梳理了一下：

<!--more-->

## 数据集构造

根据Transformer Decoder-Only特点，直接将多轮对话拼成一条session样本，过一次前向传播，得到多条回复的loss：

![sample](https://raw.githubusercontent.com/TransformersWsz/picx-images-hosting/master/image.8hg8njrad1.png)

而以往的方法是将多轮对话拆成多条样本，存在大量重复计算问题，效率低下。且该方法对于靠前轮次对话影响权重更大，不符合对话常识，靠后轮次应该权重更大，证明见：[大模型微调样本构造trick](https://zhuanlan.zhihu.com/p/641562439)

## 生成
在[karpathy/minGPT](https://github.com/karpathy/minGPT/blob/37baab71b9abea1b76ab957409a1cc2fbfba8a26/mingpt/model.py#L289)项目中，是直接粗暴地生成固定长度的文本。这样做的问题就是生成的文本无法判断何处截断。

在构造模型输入的时候，我们就加入了 `<EOS>` token，来标记文本的结束。那么在推理阶段，[如果碰到该token，则结束生成](https://github.com/TransformersWsz/GPT2-NewsTitle/blob/1e04fc50429ac767aa81b62865d41c506191a478/generate_title.py#L142)：
```python
if token == "<EOS>":
    break
```

___

## 参考
- [GPT2LMHeadModel](https://huggingface.co/docs/transformers/model_doc/gpt2#transformers.GPT2LMHeadModel)
- [mingpt](https://github.com/karpathy/minGPT/blob/master/mingpt/model.py)
- [GPT2-NewsTitle](https://github.dev/TransformersWsz/GPT2-NewsTitle)
- [全栈大模型微调框架LLaMA Factory：从预训练到RLHF的高效实现](https://www.bilibili.com/video/BV1Gt421L7dt/?spm_id_from=333.1007.tianma.23-1-87.click&vd_source=3f2411263f367ccf993c28b58688c0e7)