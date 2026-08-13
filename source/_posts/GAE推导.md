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

---

## 优势函数

数学形式为：
$$
A^\pi(s_t,a_t)=Q^\pi(s_t,a_t)-V^\pi(s_t)
$$

- $Q^\pi(s_t,a_t)$ 在状态$s$采取动作$a$后，未来能得到的期望回报
- $V^\pi(s_t)$ 在状态$s$下，按当前策略平均行动的期望回报

它表示：
> 在某个状态$s$下，采取某个动作$a$，比“这个状态下平均水平”好多少。

## TD误差

数学形式为：
$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

- $\delta_t > 0$，说明当前动作比平均值更好
- $\delta_t < 0$，说明当前动作比平均值更差

它表示：
> 当前价值估计$V(s_t)$与“一步之后的奖励 + 下一状态价值”之间的差异。


## n-step 优势估计

先定义 $n$ 步回报：

$$
G_t^{(n)} = r_t + \gamma r_{t+1} + \cdots + \gamma^{n-1}r_{t+n-1} + \gamma^n V(s_{t+n})
$$

对应的 n-step 优势估计：

$$
A_t^{(n)} = G_t^{(n)} - V(s_t)
$$

把它展开：

$$
A_t^{(n)}
= r_t + \gamma r_{t+1} + \cdots + \gamma^{n-1}r_{t+n-1} + \gamma^n V(s_{t+n}) - V(s_t)
$$

---

# 5. 把 n-step 优势写成 TD 误差和

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

> **n-step 优势 = 前 n 个 TD 误差的折扣和**

---

# 6. GAE 的核心定义

GAE 就是在所有 n-step 优势的基础上，再做一个指数加权平均：

$$
\hat A_t^{GAE(\gamma,\lambda)} =
(1-\lambda)\sum_{n=1}^{\infty}\lambda^{n-1} A_t^{(n)}
$$

其中：

- \(\lambda \in [0,1]\)
- \((1-\lambda)\lambda^{n-1}\) 是权重
- \(n\) 越大，权重越小（如果 \(\lambda<1\)）

这个定义的直觉是：

- \(n=1\)：偏向短期，方差小，偏差大
- \(n\) 大：偏向长期，偏差小，方差大
- 用 \(\lambda\) 把它们折中起来

---

# 7. 把 GAE 进一步化成 TD 误差形式

因为：

$$
A_t^{(n)}=\sum_{l=0}^{n-1}\gamma^l\delta_{t+l}
$$

代入 GAE 定义：

$$
\hat A_t^{GAE(\gamma,\lambda)}
=
(1-\lambda)\sum_{n=1}^{\infty}\lambda^{n-1}
\sum_{l=0}^{n-1}\gamma^l\delta_{t+l}
$$

交换求和顺序后，可以化简为经典形式：

$$
\hat A_t^{GAE(\gamma,\lambda)}
=
\sum_{l=0}^{\infty}(\gamma\lambda)^l \delta_{t+l}
$$

这就是最常见的 GAE 公式。

---

# 8. 最终公式

所以 GAE 的标准表达式是：

$$
\boxed{
\hat A_t^{GAE(\gamma,\lambda)}
=
\sum_{l=0}^{\infty}(\gamma\lambda)^l
\left(r_{t+l}+\gamma V(s_{t+l+1})-V(s_{t+l})\right)
}
$$

也可以写成：

$$
\boxed{
\hat A_t^{GAE(\gamma,\lambda)}
=
\sum_{l=0}^{\infty}(\gamma\lambda)^l \delta_{t+l}
}
$$

其中：

$$
\delta_t = r_t+\gamma V(s_{t+1})-V(s_t)
$$

---

# 9. 极端情况的理解

GAE 之所以好，是因为它能退化到两个经典估计方式：

## 当 \(\lambda = 0\) 时
\[
\hat A_t = \delta_t
\]

也就是**单步 TD 误差**。

- 方差小
- 偏差大

---

## 当 \(\lambda = 1\) 时
\[
\hat A_t = \sum_{l=0}^{\infty}\gamma^l\delta_{t+l}
\]

这相当于接近**Monte Carlo 优势估计**。

- 偏差小
- 方差大

---

所以 \(\lambda\) 的作用就是：

> 在 TD 的低方差和 Monte Carlo 的低偏差之间做平衡。

---

# 10. 为什么这适合 PPO？

PPO 需要一个稳定的优势估计，否则策略更新会抖动很厉害。  
GAE 的好处是：

- 既利用了多步未来信息
- 又不会像纯 Monte Carlo 那么噪声大
- 在实践中效果非常稳定

所以 PPO 基本上都会搭配 GAE 使用。

---

# 11. 一句话总结推导思路

GAE 的推导路线是：

1. 从优势函数 \(A=Q-V\) 出发  
2. 用 n-step 回报近似优势  
3. 把 n-step 优势写成 TD 误差和  
4. 再对所有 n-step 结果做指数加权平均  
5. 得到最终形式：

\[
\hat A_t^{GAE}=\sum_{l=0}^{\infty}(\gamma\lambda)^l\delta_{t+l}
\]

---

如果你愿意，我下一步可以继续给你：

1. **把 GAE 的推导写成更严格的数学推导版**
2. **举一个具体数值例子手算 GAE**
3. **解释为什么 \((1-\lambda)\sum \lambda^{n-1}A^{(n)}\) 会变成 \(\sum(\gamma\lambda)^l\delta_{t+l}\)**

如果你想，我建议下一步看 **“带数字例子手算 GAE”**，会特别直观。