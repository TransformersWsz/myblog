---
title: tf.nn.sigmoid_cross_entropy_with_logits & tf.losses.log_loss
mathjax: true
toc: true
date: 2026-07-22 00:58:36
updated: 2026-07-22 00:58:36
categories:
- Machine Learning
tags:
- TensorFlow
---
在训练模型的过程中，使用`tf.losses.log_loss`计算二分类loss，随着训练的深入，loss值为`NaN`，最终模型训练失败。排查下来发现是该函数存在数值不稳定的情况。

<!--more-->

假设DNN模型输出的$loggits=x$，概率$p=\sigma(x)$，标签$label=y$

## `tf.losses.log_loss`原理
$$
Loss = ylog(p+\epsilon)+(1-y)log(1-p+\epsilon)
$$

计算过程非常直接，当$p$极大或者极小时，float32下会精确等于1或者0，$log(0)$就会导致`NaN/Inf`，尽管$\epsilon$会做兜底，但梯度信息已经严重失真。

## `tf.nn.sigmoid_cross_entropy_with_logits`原理

$$
\begin{aligned}
Loss &= -ylog(\frac{1}{1+e^{-x}})-(1-y)(log(1-\frac{1}{1+e^{-x}})) \\
&= -xy + log(1+e^x) \\
先验\ log(1+e^x) &= max(x,0)+log(1+e^{-|x|}) \\
Loss &= max(x,0) - xy + log(1+e^{-|x|})
\end{aligned}
$$

上述公式通过两个trick确保了数值计算的稳定：
1. 留`1`法：$log(1+z)$保证最小值为0，而不是-inf
2. `max`拆分：由于$e^x$也会导致值溢出，将其转换成`max`和$e^{-|x|}$，保证极值情况下$e^{-|x|}\approx0$
