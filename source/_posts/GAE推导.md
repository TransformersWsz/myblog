---
title: GAE推导
mathjax: true
toc: true
date: 2026-08-14 01:24:09
updated: 2026-08-14 01:24:09
categories:
- Reinforcement Learning
tags:
- Generalized Advantage Estimation
---

GAE(Generalized Advantage Estimation)的推导核心：把优势函数写成一系列TD误差的加权和，再用一个参数 $\lambda$ 去平衡偏差和方差。

<!--more-->

## 1. 优势函数

数学形式为：
$$
A^\pi(s_t,a_t)=Q^\pi(s_t,a_t)-V^\pi(s_t)
$$

- $Q^\pi(s_t,a_t)$ 在状态$s$采取动作$a$后，未来能得到的期望回报
- $V^\pi(s_t)$ 在状态$s$下，按当前策略平均行动的期望回报

它表示：
> 在某个状态$s$下，采取某个动作$a$，比“这个状态下平均水平”好多少。

## 2. TD误差

数学形式为：
$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

- $\delta_t > 0$，说明当前动作比平均值更好
- $\delta_t < 0$，说明当前动作比平均值更差

它表示：
> 当前价值估计$V(s_t)$与“一步之后的奖励 + 下一状态价值”之间的差异。


## 3. 第 $n$-step 优势估计

先定义第 $n$ 步回报：

$$
G_t^{(n)} = r_t + \gamma r_{t+1} + \cdots + \gamma^{n-1}r_{t+n-1} + \gamma^n V(s_{t+n})
$$

对应的第 $n$-step 优势估计：

$$
\begin{aligned}
    A_t^{(n)} &= G_t^{(n)} - V(s_t) \\
    A_t^{(n)} &= r_t + \gamma r_{t+1} + \cdots + \gamma^{n-1}r_{t+n-1} + \gamma^n V(s_{t+n}) - V(s_t)
\end{aligned}
$$

## 4. 把第 $n$-step 优势写成 TD 误差和

我们把几项加减凑出来：
$$
\begin{aligned}
    \delta_t &= r_t + \gamma V(s_{t+1}) - V(s_t) \\

    \delta_{t+1} &= r_{t+1} + \gamma V(s_{t+2}) - V(s_{t+1}) \\

    \delta_{t+2} &= r_{t+2} + \gamma V(s_{t+3}) - V(s_{t+2}) \\
    
    &\dots \\

    \delta_{t+n-1} &= r_{t+n-1} + \gamma V(s_{t+n}) - V(s_{t+n-1})
\end{aligned}
$$

将前 $n$ 项按折扣 $\gamma$ 加权求和：

$$
\delta_t + \gamma \delta_{t+1} + \cdots + \gamma^{n-1}\delta_{t+n-1} = r_t + \gamma r_{t+1} + \cdots + \gamma^{n-1}r_{t+n-1} + \gamma^n V(s_{t+n}) - V(s_t)
$$

也就是：
$$
A_t^{(n)}=\sum_{l=0}^{n-1}\gamma^l \delta_{t+l}
$$

这一步非常重要：

> 第 $n$-step 优势 = 前 $n$ 个 TD 误差的折扣和


## 5. GAE 的核心定义
GAE将这些不同步数的优势估计进行指数加权平均：

$$
\begin{aligned}

A_t^{GAE} &= (1-\lambda)(A_t^{(1)} + \lambda A_t^{(2)} + \lambda^2 A_t^{(3)} + \cdots) \\

&= (1-\lambda)(\delta_t + \lambda(\delta_t + \gamma\delta_{t+1}) + \lambda^2(\delta_t + \gamma\delta_{t+1} + \gamma^2\delta_{t+2}) + \cdots) \\

&= (1-\lambda)(\delta_t(1+\lambda+\lambda^2+\cdots) + \gamma\delta_{t+1}(\lambda+\lambda^2+\lambda^3+\cdots) + \gamma^2\delta_{t+2}(\lambda^2+\lambda^3+\lambda^4+\cdots) + \cdots) \\

&= (1-\lambda)\left(\delta_t\frac{1}{1-\lambda} + \gamma\delta_{t+1}\frac{\lambda}{1-\lambda} + \gamma^2\delta_{t+2}\frac{\lambda^2}{1-\lambda} + \cdots\right) \\

&= \sum_{l=0}^{n-1}(\gamma\lambda)^l \delta_{t+l}

\end{aligned}
$$

这样就可以算出当前第 $t$ 步的平均优势，即可GAE。其中 $\lambda \in [0,1]$ 是在GAE中额外引入的一个超参数：
- 当 $\lambda=0$，则 $A_t^{GAE} = \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$，即是仅仅只看一步TD差分得到的优势。`方差小，偏差大`
- 当 $\lambda=1$，则 $A_t^{GAE} = \sum_{l=0}^{n-1}\gamma^l \delta_{t+l} = \sum_{l=0}^{n-1}\gamma^l r_{t+l} + \gamma^{n-1} V(s_{t+n}) - V(s_t)$，则是看每一步TD差分得到优势的完全平均值。`偏差小，方差大`


___

## 总结

GAE的推导路线是：

1. 从优势函数的定义出发  
2. 计算出第 $n$-step的优势
3. 将第 $n$-step 优势函数转换成TD误差和
4. 对每个step的优势函数做指数加权平均  
5. 得到最终的GAE公式
