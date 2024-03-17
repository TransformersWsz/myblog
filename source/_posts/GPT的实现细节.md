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
那么在构造训练集时，可以同时构造多个输入：
```text
input_1=who is your favorite basketball player? Of
input_2=who is your favorite basketball player? Of course
......
input_n-1=who is your favorite basketball player? Of course Kobe Bryant!
input_n=who is your favorite basketball player? Of course Kobe Bryant! <EOS>
```
由于训练任务是下一个单词预测，所以 $x=input[:-1], y=input[1:]$

## loss
$x$是模型可见已知的，[需要mask掉，不算入loss](https://github.com/karpathy/minGPT/blob/37baab71b9abea1b76ab957409a1cc2fbfba8a26/demo.ipynb)：
```python
y[:-1] = -1
```