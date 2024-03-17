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
原始数据集schema：
```text
input=who is your favorite basketball player?
output=Of course Kobe Bryant!
```
那么在构造训练集时，根据chunk size构造多个输入：
```text
input_1=who is your favorite basketball player? Of
input_2=who is your favorite basketball player? Of course
......
input_n-1=who is your favorite basketball player? Of course Kobe Bryant!
input_n=who is your favorite basketball player? Of course Kobe Bryant! <EOS>
```
由于训练任务是下一个单词预测，所以 $x=input[:-1], y=input[1:]$

## loss
$x$是模型可见已知的，[需要mask掉](https://github.com/karpathy/minGPT/blob/37baab71b9abea1b76ab957409a1cc2fbfba8a26/demo.ipynb)，[不算入loss](https://github.com/karpathy/minGPT/blob/37baab71b9abea1b76ab957409a1cc2fbfba8a26/mingpt/model.py#L278)：
```python
y[:-1] = -1
loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
```

## 生成
在[karpathy/minGPT](https://github.com/karpathy/minGPT/blob/37baab71b9abea1b76ab957409a1cc2fbfba8a26/mingpt/model.py#L289)项目中，是直接粗暴地生成固定长度的文本。这样做的问题就是生成的文本无法判断何处阶段。

在构造模型输入的时候，我们就加入了 `<EOS>` token，来标记文本的结束。那么在推理阶段，[如果碰到该token，则结束生成](https://github.com/TransformersWsz/GPT2-NewsTitle/blob/1e04fc50429ac767aa81b62865d41c506191a478/generate_title.py#L142)：
```python
if token == "<EOS>":
    break
```

___

## 参考
- [mingpt](https://github.com/karpathy/minGPT/blob/master/mingpt/model.py)
- [GPT2-NewsTitle](https://github.dev/TransformersWsz/GPT2-NewsTitle)