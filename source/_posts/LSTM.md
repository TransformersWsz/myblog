---
title: LSTM
mathjax: true
date: 2019-07-07 21:35:09
updated: 2019-07-07 21:35:09
categories:
- Algorithm
tags:
- RNN
- 面试
---
记录一下LSTM的模型结构与原理。

<!--more-->
## Overview
{% asset_img 1.jpeg %}

下面详细介绍LSTM的三个门：

### 遗忘门
遗忘门决定了上一时刻的单元状态 $c_{t-1}$ 有多少保留到当前时刻 $c_t$ 。

{% asset_img 2.jpeg %}

公式如下：
$$
f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)
$$

### 输入门
输入门决定了当前时刻网络的输入 $x_t$ 有多少保存到单元状态 $c_t$ 。

{% asset_img 3.jpeg %}

公式如下：
$$
i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)
$$

### 当前输入的单元状态 $\tilde{c_{t}}$
可以把它想象成不包含上一时刻的长期状态 $c_{t-1}$ 时，我们生成的当前此刻的长期状态 $\tilde{c_{t}}$ 。

{% asset_img 4.jpeg %}

公式如下：
$$
\tilde{c}_{t}=\tanh \left(W_{c} \cdot\left[h_{t-1}, x_{t}\right]+b_{c}\right)
$$

### 计算当前时刻的单元状态 $c_t$

{% asset_img 5.jpeg %}

公式如下：
$$
c_{t}=f_{t} \circ c_{t-1}+i_{t} \circ \tilde{c}_{t}
$$

这样，我们就能够将当前的记忆 $\tilde{c_{t}}$ 和长期的记忆 $c_{t-1}$ 组合在一起，形成新的单元状态 $c_t$ 。由于遗忘门的控制，它可以保存很久之前的信息，由于输入门的控制，它又可以避免当前无关紧要的内容进入记忆。

### 输出门
用来控制单元状态 $c_t$ 有多少输入到 LSTM 的当前输出值 $h_t$ 。

{% asset_img 6.jpeg %}

公式如下：
$$
o_{t}=\sigma\left(W_{o} \cdot\left[h_{t-1}, x_{t}\right]+b_{o}\right)
$$


### 计算输出值 $h_t$

{% asset_img 7.jpeg %}

公式如下：
$$
h_{t}=o_{t} \circ \tanh \left(c_{t}\right)
$$
___
## 参考
[LSTM：RNN最常用的变体](https://zhuanlan.zhihu.com/p/44124492)