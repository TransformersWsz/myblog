---
title: GLM
mathjax: true
date: 2019-05-29 21:55:37
updated: 2019-05-29 21:55:37
categories: 
- Machine Learning
tags:
- Neural Networks
- Algorithm
- Linear Regression
---

##  为什么要引入GLM？

我们知道了”回归“一般是用于预测样本的值，这个值通常是连续的。但是受限于其连续的特性，一般用它来进行分类的效果往往很不理想。为了保留线性回归”简单效果又不错“的特点，又想让它能够进行分类，因此需要对预测值再做一次处理。这个多出来的处理过程，就是GLM所做的最主要的事。而处理过程的这个函数，我们把它叫做连接函数。

<!--more-->

如下图是一个广义模型的流程：

{% asset_img 1.jpg %}

图中，当一个处理样本的回归模型是线性模型，且连接函数满足一定特性（特性下面说明）时，我们把模型叫做广义线性模型。因为广义模型的最后输出可以为离散，也可以为连续，因此，用广义模型进行分类、回归都是可以的。

但是为什么线性回归是广义线性模型的子类呢，因为连接函数是 $f(x)=x$ 本身的时候，也就是不做任何处理时，它其实就是一个线性回归。

所以模型的问题就转化成获得合适的连接函数？以及有了连接函数，怎么求其预测函数 $h_\theta(x)$ ？

## 定义

刚才说了，只有连接函数满足一定特性才属于广义线性模型。特性是什么呢？先简单描述下背景。

在广义线性模型中，为了提高可操作性，因此限定了概率分布必须满足指数族分布：
$$
p(y;\eta) = b(y)e^{ {\eta^T}{T(y)-a(\eta)} }
$$

> - $\eta$ 称为这个分布的 **自然参数(natural parameter)** 或者 **规范参数(canonical parameter)**。$\eta=\theta^TX$ ，即自然参数=参数与自变量X的线性组合。
> - $T(y)$ 称为 **充分统计量(sufficient statistic)**。
> - $a(\eta)$ 称为 **对数分割函数(log partition function)**，$e^{-a(\eta)}$ 是分布 $p(y;\eta)$ 的归一化常数，用来确保该分布对 $y$ 的积分为 1。
> - 当 $T,a,b$ 固定之后，也就确定了这样一个以 $\eta$ 为参数的分布族。

### 广义线性模型的三个假设

- $(y|x;\theta)\sim Exponential Family(\eta)$：给定样本 $x$ 和 参数 $\theta$ ，样本分类 $y$ 服从指数分布。
- 给定一个 $x$ ，我们需要的目标函数为 $h_\theta(x) = E[T(y) | x]$ 。
- $\eta = (\vec \theta)^T \vec X$ ，即自然参数 $\eta$ 和 输入 $\vec X$ 满足线性关系。

## 连接函数的获取

从上图可以看到 $\eta$ 为函数的输入，而 $h_\theta(x)$ 为函数的输出，所以有公式：
$$
h_\theta(x) = f(\eta)
$$
但是我们会把 $f$ 的逆 $f^{-1}$ 称为**连接函数** ， 也即以 $h_\theta(x)$ 为自变量，$\eta$ 为因变量的函数为连接函数：
$$
\eta = f^{-1}(h_\theta(x))
$$
所以求连接函数的步骤也就变成：

1. 将 $\vec Y$ 、$\vec X$ 所满足的分布转换成指数分布形式。
2. 在指数分布形式中获得 $T(y)$ 的函数形式和 $\eta$  的值。
3. 算出 $E[T(y)|x]$ 和 $\eta$ 的关系，并把 $(\vec \theta)^T$ 代入到$\eta$ 中，获得连接函数。

## 常见连接函数求解及对应回归

### 伯努利分布 ---> Logistic 回归

伯努利分布只有0、1两种情况，因此它的概率分布可以写成：
$$
p(y;\phi) = \phi^y(1-\phi)^{1-y} \qquad y=[0,1] \qquad \phi: 实验为1发生的概率
$$
下面是逻辑回归的推导过程：

{% asset_img 2.png %}

### 多项分布 ---> softmax 回归

前面说过的分类问题都是处理那些分两类的问题。比如区分猫或者狗的问题，就是一类是或者否的问题。但是现实生活中还有更加多的多类问题。比如猫分类，有田园猫，布偶猫，暹罗猫各种猫，这里就不能够用两分类来做了。 

这里先设问题需要区分 $k$ 类，即 $y \in \lbrace1, 2, 3, ..., k\rbrace$ 。此处无疑使用多项式来建模。多项式分布是二项分布的一个扩展，其取值可以从$\lbrace1, 2, 3, ..., k\rbrace$ 中取，可以简单建模如下：
$$
\left[
\begin{matrix}
 1      & 0      & \cdots & 0      \\
 0      & 1      & \cdots & 0      \\
 \vdots & \vdots & \ddots & \vdots \\
 0      & 0      & \cdots & 1      \\
\end{matrix}
\right]
$$
例如当$y=2$ 时，第二个数为1，其它数为0，因此它的概率分布如下：

{% asset_img 3.png %}

___

## 参考

- [机器学习笔记五：广义线性模型（GLM）](https://blog.csdn.net/xierhacker/article/details/53364408)
- [Softmax回归（Softmax Regression）](https://www.cnblogs.com/BYRans/p/4905420.html)
- [机器学习之回归（二）：广义线性模型（GLM）](https://cloud.tencent.com/developer/article/1005793)