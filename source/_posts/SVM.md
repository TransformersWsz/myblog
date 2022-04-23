---
title: SVM
mathjax: true
date: 2019-06-22 22:01:08
categories: 
- Machine Learning
tags:
- Algorithm
- Classification
---


支持向量机(Support Vector Machine)是一种二分类模型。它的基本模型是定义在特征空间上的间隔最大的线性分类器。SVM的学习算法是求解凸二次规划的最优化算法。

<!--more-->

假设在一个二维线性可分的数据集中，图一A所示，我们要找到一条把两组数据分开，这条直线可以是图一B中的直线，也可以是图一C中的直线，或者图一D中的直线，但是哪条直线能够达到最好的泛化能力呢？那就是一个能使两类之间的空间大小最大的一个超平面。

{% asset_img 1.png %}

这个超平面在二维平面上看到的就是一条直线，在三维空间中就是一个平面。因此，我们把这个划分数据的决策边界统称为超平面。<font color="green">离这个超平面最近的点就叫做支持向量，点到超平面的距离叫间隔。</font>支持向量机就是要使超平面和支持向量之间的间隔尽可能的大，这样超平面才可以将两类样本准确的分开，而保证间隔尽可能的大就是保证我们的分类器误差尽可能的小，尽可能的健壮。

# 线性可分SVM

要使得支持向量到超平面的间隔最大化，我们首先定义超平面 $h(x)$ ：
$$
h(x) = w^Tx + b \qquad w为权重向量，b为偏置向量
$$
样本点 $x$ 到最优超平面的几何间隔为：
$$
r = \frac {h(x)} {||w||} = \frac {w^T + b} {||w||}
$$
$||w||$ 是向量 $w$ 的内积，即 $||w|| = \sqrt{w_0^2 + w_1^2 + \dots +  w_n^2 }$ ，而 $h(x)$ 表示函数间隔：
$$
\hat{r} = h(x)
$$
函数间隔 $h(x)$ 不是一个标准的间隔度量，它不适合用来做最大化的间隔值。因为，一旦超平面固定以后，如果我们人为的放大或缩小 $w$ 和 $b$ 值，那这个超平面也会无限的放大或缩小，这将对分类造成严重影响。而几何间隔是函数间隔除以 $w$ ，当 $w$ 的值无限放大或缩小时，$||w||$ 也会等倍地放大或缩小，而整个 $r$ 保持不变，它只随着超平面的移动而变化，不受两个参数的影响。因而，我们用几何间隔来做最大化间隔度量。

## 最大化间隔

在SVM中，我们把几何间隔 $r$ 作为最大化间隔，并且采用 $-1$ 和 $+1$ 作为类别标签。

如下图所示，在这个 $\mathbb{R}^2$ 空间中，假设我们已经确定了一个超平面（图中虚线），这个超平面的函数关系式为 $h(x) = w^Tx + b = 0$ 。我们想要所有的样本点都尽可能的原理这个超平面，只需保证支持向量的点 $x^*$ 尽可能地远离它。

{% asset_img 2.png %}

我们把其中一个支持向量 $x^*$ 到最优超平面的距离定义为：
$$
r^* = \frac {h(x^*)} {||w||} = 
\begin{cases}
\frac {1} {||w||}&  \quad {if : y* = h(x^*) = +1}\\
\frac {-1} {||w||}& \quad {if : y* = h(x^*) = -1}
\end{cases}
$$
这是我们通过把函数间隔 $h(x)$ 固定为 $1$ 而得来的。我们可以把这个式子想象成还存在两个平面，这两个平面分别是 $w^Tx_s+b=1$ 和 $w^Tx_s+b=-1$ ，对应上图中的两根实线。这些支持向量 $x_s$ 就在这两个平面上，这两个平面离最优超平面的距离越大，我们的间隔也就越大。对于其他的点 $x_i$ 如果满足 $w^Tx_i+b>1$ ，则被分为 $1$ 类，如果满足满足 $w^Tx_i+b<-1$ ，则被分为 $-1$ 类。即有约束条件：

$$
\begin{cases}
w^Tx_i+b \geqslant 1 &  \quad y_i = +1 \\
w^Tx_i+b \leqslant -1 & \quad y_i = -1
\end{cases}
$$
支持向量到超平面的距离知道后，那么分割的间隔 $\gamma$ 为：
$$
\gamma = 2r^* = \frac {2} {||w||}
$$
注意到最大化 $\frac {2} {||w||}$ 和最小化 $\frac {1} {2} ||w||^2$ 是等价的，于是就得到下面的线性可分支持向量机学习的最优化问题：
$$
\begin{cases}
\min_{w, b} \; \frac {1} {2} \|w\|^2 \\
y_i(w^Tx_i+b) \geqslant 1, \quad (i = 1, \dots , n) 
\end{cases}
$$

这种式子采用拉格朗日乘数法来求解，即：
$$
L(x) = f(x) + \sum \alpha g(x)
$$
$f(x)$ 是我们需要最小化的目标函数， $g(x)$ 是不等式约束条件， $\alpha$ 是对应的约束系数，也称拉格朗日乘子。为了使得拉格朗日函数得到最优解，我们需要加入能使该函数有最优化解法的KKT条件，或者叫最优化条件。即假设存在一点 $x^*$：

- $L(x^{\star})$ 对 $x^{\star}$ 求导为 $0$
- $\alpha_ig_i(x^*)=0$ 对于所有的 $i = 1,\dots,n$

这样构造的拉格朗日函数为：
$$
L(w, b, a) = \frac {1} {2} w^Tw - \sum_{i=1}^n a_i[y_i(w^T x_i + b) - 1]
$$
以上的KKT条件 $\alpha_i[y_i(w^Tx_i+b)-1] = 0$ 表示，只有距离最优超平面的支持向量 $(x_i, y_i)$ 对应的 $\alpha$ 非零，其他所有点集的 $\alpha$ 等于零。综上所述，引入拉格朗日乘子后，我们的目标变为：
$$
\min_{w,b}\max_{a \geqslant 0} L(w, b, a)
$$
根据拉格朗日对偶性，目标问题的对偶问题是极大极小问题，即先求 $L(w, b, \alpha)$ 对 $w, b$ 的极小，再求对 $\alpha$ 的极大：
$$
\max_{a \geqslant 0} \min_{w, b} L(w, b, \alpha)
$$
用 $L(w, b, \alpha)$ 对 $w$ 和 $b$ 分别求导，并令其等于 $0$ ：
$$
\begin{cases}
\frac {\partial L(w, b, \alpha)} {\partial w} = 0\\
\frac {\partial L(w, b, \alpha)} {\partial b} = 0
\end{cases}
$$
得到：
$$
\begin{cases}
w = \sum_{i=1}^n \alpha_i y_i x_i \\
\sum_{i=1}^n \alpha_i y_i = 0
\end{cases}
$$
把该式代入原来的拉格朗日式子得（推导见《统计学习方法》P103~P105）：
$$
L(\alpha) = \sum_{i=1}^n \alpha_i - \frac {1} {2} \sum_{i=1}^n \sum_{j=1}^n \alpha_i \alpha_j y_i y_j x_i^T x_j \\
\sum_{i=1}^n \alpha_i y_i = 0, \quad \alpha_i \geqslant 0 (i = 1, \dots, n)
$$
该 $L(\alpha)$ 函数消去了向量 $w$ 和向量 $b$ ，仅剩 $\alpha$ 这个未知参数，只要我们能够最大化 $L(\alpha)$，就能求出对应的 $\alpha$ ，进而求得  $w$ 和 $b$ 。对于如何求解 $\alpha$，SMO算法给出了完美的解决方案，下一节我们详细讲述。这里我们假设通过SMO算法确定了最优 $\alpha^*$，则：
$$
w^* = \sum_{i=1}^n \alpha_i^* y_i x_i
$$
最后使用一个正的支持向量 $x^*$ ，就可以计算出 $b$ ：
$$
b^* = 1 - w^{*T} x^*
$$

## <a id="softgap">软间隔</a>

以上的推导都是在<font color="red">完全线性可分</font>的条件下进行的，但是现实世界的许多问题并不都是线性可分的，尤其存在许多复杂的非线性可分的情形。要解决这些线性不可分问题，有如下两种方法：

- 放宽严格的间隔，构造[软间隔](#softgap)。
- 运用[核函数](#kernel)将数据映射到高维空间，然后在高维空间中寻找超平面进行线性划分。

我们首先讨论软间隔。假设两个类有几个数据点混在一起，这些点对最优超平面形成了噪声干扰，软间隔就是要扩展一下我们的目标函数和KKT条件，允许少量这样的噪声存在。具体地说，就要引入松驰变量 $\xi$ 来量化分类器的违规行为：
$$
\begin{cases}
\min_{w, b} \; \frac {1} {2} \|w\|^2 + C\sum_{i=1}^n \xi_i, \quad C为惩罚因子 \\
y_i (w^T x_i + b) \geqslant 1 - \xi_i , \quad i = 1,\dots, n \\
\xi_i \geqslant 0 ,  \quad i = 1, \dots , n
\end{cases}
$$


### $C$ 和 $\xi$

{% asset_img 3.jpg %}

如上图所示，$\xi$ 表示噪声样本点到本类样本点边界的偏移距离。$C$ 可被视为一个由用户依据经验或分析选定的“正则化”参数。噪声点在现实世界是天然存在的，如果对于他们不进行容错，那么我们是无论如何也不能把样本分开的。而引入惩罚因子，目的就是，对这类误分的样本进行容错，相当于把点拉到正确一侧：

- 当 $C$ 很大时，$\xi$ 趋近于0，表示惩罚很大，容忍度很低。这样错分较少，对样本的拟合性较好，但容易过拟合。
- 当 $C$ 很小时，$\xi$ 变大，表示惩罚很小，容忍度高。这样错分较多，对样本的拟合性下降。

对上述不等式同样运用拉格朗日乘子法和KKT条件得（推导见《统计学习方法》P109~P111）：
$$
L(\alpha) = \sum_{i=1}^n \alpha_i - \frac {1} {2} \sum_{i=1}^n \sum_{j=1}^n \alpha_i \alpha_j y_i y_j x_i^T x_j \\
\sum_{i=1}^n \alpha_i y_i = 0, \quad 0 \leqslant \alpha_i \leqslant C (i = 1, \dots , n)
$$
可以看到，松驰变量 $\xi$ 没有出现在 $L(\alpha)$ 中，线性可分与不可分的差异体现在约束 $\alpha_i \geqslant 0$ 被替换成了约束 $0 \leqslant \alpha_i \leqslant C$。但是，这两种情况下求解 $w$ 和 $b$ 是非常相似的，对于支持向量的定义也都是一致的。在不可分情况下，对应的KKT条件为：
$$
\alpha_i[y_i(w^Tx_i + b) - 1 + \xi_i] = 0, \quad (i = 1, \dots, n)
$$

## SMO算法

1996年， John Platt发布了一个称为SMO的强大算法，用于训练SVM。 SMO表示序列最小优化（Sequential Minimal Optimization）。 Platt的SMO算法是将大优化问题分解为多个小优化问题来求解，这些小优化问题往往很容易求解，并且对它们进行顺序求解的结果与将它们作为整体来求解的结果是完全一致的。

### 目标

求出一系列 $\alpha$，一旦求出了这些  $\alpha$，就很容易计算出权重向量 $w$ 和 $b$，并得到分隔超平面。

### 工作原理

每次循环中选择两个 $\alpha$ 进行优化处理。一旦找到一对合适的 $\alpha$ ，那么就增大其中一个同时减小另一个。这里所谓的“合适”就是指两个 $\alpha$ 必须要符合一定的条件，条件之一就是这两个 $\alpha$ 必须要在间隔边界之外，而其第二个条件则是这两个  $\alpha$ 还没有进行过区间化处理或者不在边界上。
对SMO具体的分析如下，在上一节我们已经得出了：
$$
L(\alpha) = \sum_{i=1}^n \alpha_i - \frac {1} {2} \sum_{i=1}^n \sum_{j=1}^n \alpha_i \alpha_j y_i y_j x_i^T x_j \\
\sum_{i=1}^n \alpha_i y_i = 0, \quad 0 \leqslant \alpha_i \leqslant C (i = 1, \dots , n)
$$
其中 $(x_i, y_i)$ 已知，$C$ 可以人为设定。现在就是要最大化 $L(\alpha)$ ，求得参数 $\alpha = [\alpha_1, \alpha_2, \dots, \alpha_n]$。SMO算法是一次选择两个 $\alpha$ 进行优化，那我们就选择 $\alpha_1$ 和 $\alpha_2$ ，然后把其它参数 $[\alpha_3, \alpha_4, \dots, \alpha_n]$ 固定，这样 $\alpha_1, \alpha_2$ 表示为下面的式子，其中 $\zeta$ 为实数值：
$$
\alpha_1 y_1 + \alpha_2 y_2 = - \sum_{i=3}^n \alpha_i y_i = \zeta
$$
然后用 $\alpha_2$ 来表示 $\alpha_1$ ：
$$
\alpha_1 = \frac {\zeta - \alpha_2 y_2} {y_1}
$$
把上式代入 $L(\alpha)$ 中：
$$
L(\alpha) = L(\frac {\zeta - \alpha_2 y_2} {y_1}, \alpha_2, \dots, \alpha_n)
$$
省略一系列化解过程后，最后会化解成我们熟悉的一元二次方程，$a, b, c$ 均是实数值：
$$
L(\alpha_2) = a \alpha_2^2 + b \alpha_2 + c
$$
最后对 $\alpha_2$ 求导，解得 $\alpha_2$ 的具体值，我们暂时把这个实数值叫 $\alpha_2^{\star}$ ，而这个 $\alpha_2^{\star}$ 需要满足一个条件 $L \leqslant \alpha_2^{\star} \leqslant H$ ，如下图所示：

{% asset_img 4.png %}

根据之前的条件 $0 \leqslant \alpha_i \leqslant C$ 和等式 $\alpha_1 y_1 + \alpha_2 y_2 = \zeta$ ，当 $y_1$ 和 $y_2$ 异号时：
$$
\begin{cases}
L &= \max(0, \alpha_2 - \alpha_1) \\
H &= \min(C, C + \alpha_2 - \alpha_1)
\end{cases}
$$
当 $y_1$ 和 $y_2$ 同号时：
$$
\begin{cases}
L &= \max(0, \alpha_2 + \alpha_1 - C) \\
H &= \min(C, \alpha_2 + \alpha_1)
\end{cases}
$$
最后，满足条件的 $\alpha_2$ 应该由下面的式子得到， $\alpha_2^{\star\star}$ 才为最终的值：
$$
\alpha_2^{**} = \begin{cases}
H, \quad \alpha_2^* > H \\
\alpha_2^*, \quad L \leqslant \alpha_2^* \leqslant H \\
L, \quad \alpha_2^* < L
\end{cases}
$$
求得 $\alpha_2^{\star\star}$ 后就能求得 $\alpha_1^{\star\star}$ 了，然后我们重复地按照最优化 $ (\alpha_1, \alpha_2) $ 的方式继续选择 $ (\alpha_3, \alpha_4), (\alpha_5, \alpha_6), \dots, (\alpha_{n-1}, \alpha_n) $ 进行优化求解，这样 $ \alpha = [\alpha_1, \alpha_2, \dots, \alpha_n] $ 求解出来后，整个线性划分问题就迎刃而解。

# <a id="kernel">核函数</a>

对于以上几节讲的SVC算法，我们都在线性可分或存在一些噪声点的情况下进行的二分类，但是如果我们存在两组数据，它们的散点图如下图所示，你可以看出这完全是一个非线性不可分的问题，我们无法使用之前讲的SVC算法在这个二维空间中找到一个超平面把这些数据点准确的分开。

{% asset_img 5.png %}

解决这个划分问题我们需要引入一个核函数，核函数能够恰当地计算给定数据的内积，将数据从输入空间的非线性转变到特征空间，特征空间具有更高甚至无限的维度，从而使得数据在该空间中被转换成线性可分的。如下图所示，我们把二维平面的一组数据，通过核函数映射到了一个三维空间中，这样，我们的超平面就面成了一个平面（在二维空间中是一条直线），这个平面就可以准确的把数据划分开了。

{% asset_img 6.png %}

核函数有Sigmoid核、线性核、多项式核和高斯核等，其中高斯核和多项式核比较常用，两种核函数均可以把低维数据映射到高维数据。高斯核的公式如下，$\sigma$ 是达到率，即函数值跌落到 $0$ 的速度参数：
$$
K(x_1, x_2) = e^{\frac {- \|x_1 - x_2\|^2} {2 \sigma^2}}
$$
多项式核函数的公式如下，$R$ 为实数，$d$ 为低维空间的维数：
$$
K(x_1, x_2) = (\langle x_1, x_2 \rangle + R)^d
$$
应用于我们的上个例子，我们先定义用 $\phi : x \to H$ 表示从输入空间 $x \subset \mathbb{R}^n$ 到特征空间 $H$ 的一个非线性变换。假设在特征空间中的问题是线性可分的，那么对应的最优超平面为：
$$
w^{\phi T} \phi(x) + b = 0
$$
通过拉格朗日函数我们推导出：
$$
w^{\phi *} = \sum_{i=1}^n \alpha_i^* y_i \phi(x_i)
$$
代入上式的特征空间的最优超平面为：
$$
\sum_{i=1}^n \alpha_i^* y_i \phi^T(x_i) \phi(x) + b = 0
$$
这里的 $\phi^T(x_i) \phi(x)$ 表示内积，用核函数代替内积则为：
$$
\sum_{i=1}^n \alpha_i^* y_i K(x_i, x) + b = 0
$$
我们的核函数均是内积函数，通过在低维空间对输入向量求内积来映射到高维空间，从而解决在高维空间中数据线性可分的问题。

我们以多项式核来解释一下为什么核函数可以把低维数据映射成高维数据。

假设有两个输入样本，它们均为二维行向量 $a = [x_1, x_2], b = [x_3, x_4]$ ，他们的内积为：
$$
\langle a, b \rangle = a b^T = \begin{bmatrix} x_1 & x_2 \end{bmatrix} \begin{bmatrix} x_3 \\ x_4 \end{bmatrix} = x_1 x_3 + x_2 x_4
$$
用多项式核函数进行映射，令 $R=0, d=2$：
$$
K(a, b) = (\langle x_1, x_2 \rangle)^2 = (x_1 x_3 + x_2 x_4)^2 = x_1^2 x_3^2 + 2x_1 x_2 x_3 x_4 + x_2^2 x_4^2 = \phi(a) \phi(b)
$$
按照线性代数中的标准定义， $\phi(a)$ 和 $\phi(b)$ 为映射后的三维行向量和三维列向量，即：
$$
\phi(a) = \begin{bmatrix} x_1^2 & \sqrt 2 x_1 x_2 & x_2^2 \end{bmatrix} \\
\phi(b) = \begin{bmatrix} x_3^2 \\ \sqrt2 x_3 x_4 \\ x_4^2  \end{bmatrix}
$$
它们的内积用向量的方式表示则更直观：
$$
\phi(a) \phi(b) = \begin{bmatrix} x_1^2 & \sqrt 2 x_1 x_2 & x_2^2 \end{bmatrix} \begin{bmatrix} x_3^2 \\ \sqrt2 x_3 x_4 \\ x_4^2  \end{bmatrix} = x_1^2 x_3^2 + 2x_1 x_2 x_3 x_4 + x_2^2 x_4^2
$$
这样我们就把二维数据映射成了三维数据。对于高斯核的映射，会用到泰勒展开式，这个后面再学习了。

# 损失函数

上面说了那么多，全是从数学角度进行分析推导的。你可能明白了SVM的数学原理，当你进行编程的时候，还是一脸懵逼。因为如果按照上面的求解过程来的话，实在是太复杂了。但是在计算机里，求解SVM却是非常简单的事。我们只需给出SVC的损失函数，然后使用GD算法，就能很好地求出 $\theta$ ，也就是上述的 $w$ ：
$$
J(\theta) = \min_{\theta} C [ \sum_{i=1}^m y^{(i)} cost_1(\theta^T x^{(i)}) + (1 - y^{(i)})cost_0(\theta^T x^{(i)}) ] + \frac {1} {2} \sum_{j=1}^n \theta_j^2 \quad C 为惩罚因子
$$
上述函数分析具体可见 https://github.com/TransformersWsz/Halfrost-Field/blob/master/contents/Machine_Learning/Support_Vector_Machines.ipynb

___

## 参考

- [SVM原理以及Tensorflow 实现SVM分类(附代码) ](https://www.cnblogs.com/vipyoumay/p/7560061.html?tdsourcetag=s_pcqq_aiomsg)
- [深度讲解支持向量机背后的数学思想](https://baijiahao.baidu.com/s?id=1621964725382082396&wfr=spider&for=pc)
- [Support_Vector_Machines.ipynb](https://github.com/TransformersWsz/Halfrost-Field/blob/master/contents/Machine_Learning/Support_Vector_Machines.ipynb)
- 《统计学习方法》
- [吴恩达机器学习](https://study.163.com/course/courseLearn.htm?courseId=1004570029#/learn/video?lessonId=1052089362&courseId=1004570029)