---
title: EOM公式推导
mathjax: true
toc: true
date: 2025-10-10 21:45:29
updated: 2025-10-10 21:45:29
categories:
- Marketing
tags:
- EOM
---
在uplift建模中，除了AUUC、QINI指标，还有EOM。它是基于离线RCT模拟评估在线业务收益的指标，EOM越高，业务收益越高。

<!--more-->

在这里记录下EOM的公式推导。


![EOM](https://github.com/TransformersWsz/picx-images-hosting/raw/master/image.1ovti3dugm.webp)

这个推导依赖于**概率论中的两个核心概念**：

1.  **全期望定律 (Law of Total Expectation)**
2.  **随机实验中的独立性 (Independence in Randomized Experiments)**

**目标：** 证明 $\text{E}[Z] = \text{E}[Y | T = h(\mathbf{X})]$

其中，随机变量 $Z$ 的定义为：
$$Z = \sum_{t=0}^{K} \frac{1}{p_t} Y \mathbb{I}\{h(\mathbf{X}) = t\} \mathbb{I}\{T = t\}$$
这里 $p_t = P(T=t)$ 是用户被分配到干预 $t$ 的概率。

### **第一步：代入 $Z$ 的定义并利用期望的线性性**

期望 $\text{E}[\cdot]$ 具有线性性，因此我们可以将期望 $\text{E}[Z]$ 拆分成和式的期望：
$$\text{E}[Z] = \text{E}\left[\sum_{t=0}^{K} \frac{1}{p_t} Y \mathbb{I}\{h(\mathbf{X}) = t\} \mathbb{I}\{T = t\}\right]$$

将求和 $\sum$ 和常量 $\frac{1}{p_t}$ 移出期望：
$$\text{E}[Z] = \sum_{t=0}^{K} \frac{1}{p_t} \text{E}\left[Y \mathbb{I}\{h(\mathbf{X}) = t\} \mathbb{I}\{T = t\}\right] \quad (*)$$

### **第二步：利用全期望定律和指示函数**

根据概率论中对随机变量乘积期望的定义，两个指示函数 $\mathbb{I}\{A\}$ 和 $\mathbb{I}\{B\}$ 相乘，等价于条件 $A$ 和 $B$ **同时成立**。

$$\text{E}[Y \mathbb{I}\{h(\mathbf{X}) = t\} \mathbb{I}\{T = t\}] = \text{E}\left[Y \cdot \mathbb{I}\left\{h(\mathbf{X}) = t, T = t\right\}\right]$$

利用全期望定律，将期望写成 **联合概率的积分** 形式：
$$\text{E}\left[Y \cdot \mathbb{I}\left\{h(\mathbf{X}) = t, T = t\right\}\right] = \text{E}\left[Y \mid h(\mathbf{X}) = t, T = t\right] \cdot P\left(h(\mathbf{X}) = t, T = t\right)$$

将这个结果代回 $(*)$：
$$\text{E}[Z] = \sum_{t=0}^{K} \frac{1}{p_t} \cdot \text{E}\left[Y \mid h(\mathbf{X}) = t, T = t\right] \cdot P\left(h(\mathbf{X}) = t, T = t\right)$$

### **第三步：利用随机实验的独立性**

在RCT样本中，用户被分配到干预组 $T$ 的过程是独立于用户的特征 $\mathbf{X}$（以及模型基于 $\mathbf{X}$ 的预测 $h(\mathbf{X})$）的。

因此，**事件 $\{h(\mathbf{X})=t\}$ 和 $\{T=t\}$ 是相互独立的。**

根据独立性，联合概率可以分解：
$$
\begin{aligned}
P\left(h(\mathbf{X}) = t, T = t\right) &= P\left(h(\mathbf{X}) = t\right) \cdot P\left(T = t\right)  \\
&= P\left(h(\mathbf{X}) = t\right) \cdot p_t
\end{aligned}
$$

$$
\begin{aligned}
\text{E}[Z] &= \sum_{t=0}^{K} \frac{1}{p_t} \cdot \text{E}\left[Y \mid h(\mathbf{X}) = t, T = t\right] \cdot \left(P\left(h(\mathbf{X}) = t\right) \cdot p_t\right) \\
&= \sum_{t=0}^{K} \text{E}\left[Y \mid h(\mathbf{X}) = t, T = t\right] \cdot P\left(h(\mathbf{X}) = t\right)
\end{aligned}
$$

回顾全期望定律：$\text{E}[A] = \sum_{i} \text{E}[A | B=b_i] P(B=b_i)$。

$$
\begin{aligned}
\text{E}[Z] &=\sum_{t=0}^{K} \text{E}\left[Y \mid h(\mathbf{X}) = t, T = t\right] \cdot P\left(h(\mathbf{X}) = t\right) \\
&=\sum_{t=0}^{K} \text{E}\left[Y \mid T = t, h(\mathbf{X}) = t\right] \cdot P\left(h(\mathbf{X}) = t\right) \\
&=  \text{E}[Y | T = h(\mathbf{X})]
\end{aligned}
$$

___

在营销场景的在线运筹中，$h(X) = t$ 表示运筹出一张券面额，如果实发面额$T$也等于$t$，那么$z=\frac{Y}{p_t}$，这样即可模拟出在线业务收益。$p_t$是该treatment的样本分布占比，$\frac{1}{p_t}$表示IPW，从而避免样本不均的影响。