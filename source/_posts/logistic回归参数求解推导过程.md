---
title: logistic回归参数求解推导过程
mathjax: true
date: 2021-03-23 22:11:58
categories:
- Machine Learning
tags:
- 面试
---
记录一下逻辑回归的参数求解推导过程：

<!--more-->

## 损失函数

线性回归的表达式为：$f(x) = wx+b$，为了消除后面的$b$，令$\theta = [w \quad b], x = [x \quad 1]^T$，则$f(x) = \theta x$

将其转换为逻辑回归模型：$y=\sigma(f({x}))=\sigma\left({\theta} {x}\right)=\frac{1}{1+e^{-{\theta} {x}}}$

我们把单个样本看作一个事件，那么这个事件发生的概率为：
$$
P(y \mid {x})=\left\{\begin{array}{r}
p, y=1 \\
1-p, y=0
\end{array}\right.
$$
它等价于：$P\left(y_{i} \mid {x}_{i}\right)=p^{y_{i}}(1-p)^{1-y_{i}}$

如果我们采集到了一组数据一共N个，$\left\{\left({x}_{1}, y_{1}\right),\left({x}_{2}, y_{2}\right),\left({x}_{3}, y_{3}\right) \ldots\left({x}_{N}, y_{N}\right)\right\},$ 这个合成在一起的合事件发生的总概率如下：
$$
\begin{aligned}
P_{total} &= P(y_1|x_1)P(y_2|x_2)P(y_3|x_3) \ldots P(y_N|x_N) \\
&= \prod_{i=1}^{N} p^{y_{i}}(1-p)^{1-y_{i}} \\
F(\theta) &= ln(P_{total}) = \sum_{i=1}^N ln(p^{y_{i}}(1-p)^{1-y_{i}}) \\
&= \sum_{i=1}^N y_ilnp + (1-y_i)ln(1-p) \\
其中 p &= \frac{1}{1+e^{-{\theta} {x}}}
\end{aligned}
$$
为了符合损失函数的含义，将其定义为：
$$
L(\theta) = -F(\theta)
$$

## 推导

$$
\frac{\partial L}{\partial \theta} = \frac{\partial L}{\partial p} \times \frac{\partial p}{\partial \theta}
$$

先求$\frac{\partial p}{\partial \theta}$ :
$$
\begin{aligned}
p' &= (\frac{1}{1+e^{-\theta x}})' \\
&= \frac{-1}{(1+e^{-\theta x})^2} \cdot e^{-\theta x} \cdot -x \\
&= \frac{1}{1+e^{-\theta x}} \cdot \frac{e^{-\theta x}}{1+e^{-\theta x}} \cdot x \\
&= p(1-p)x
\end{aligned}
$$
求$\frac{\partial L}{\partial \theta}$ :
$$
\begin{aligned}
\nabla F(\theta) &= \nabla (\sum_{i=1}^N y_ilnp + (1-y_i)ln(1-p)) \\
&= \sum_{i=1}^N y_i \frac{1}{p} p' + (1-y_i)\frac{-1}{1-p}p' \\
&= \sum_{i=1}^N y_i(1-p)x_i - (1-y_i)px_i \\
&= \sum_{i=1}^N (y_i-p) x_i \\
\end{aligned}
$$
因此 $\frac{\partial L}{\partial \theta} = \sum_{i=1}^N (p-y_i)x_i$

## 梯度更新

通过反向传播，$\theta$ 的更新过程如下：
$$
\theta := \theta - \alpha \sum_{i=1}^N (y_i - \frac{1}{1+e^{-\theta x_i}}) x_i
$$
___
## 参考
- [逻辑回归 logistics regression 公式推导](https://zhuanlan.zhihu.com/p/44591359)