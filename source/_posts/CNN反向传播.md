---
title: CNN反向传播
mathjax: true
date: 2019-06-03 22:49:21
updated: 2019-06-03 22:49:21
categories:
- Machine Learning
tags:
- Neural Networks
- Algorithm
---

深度神经网络(DNN)反向传播的公式推导可以参考之前的博客：[反向传播](https://transformerswsz.github.io/2019/05/29/%E5%8F%8D%E5%90%91%E4%BC%A0%E6%92%AD/)

<!--more-->

要套用DNN的反向传播算法到CNN，有几个问题需要解决：

- 池化层没有激活函数，我们可以令池化层的激活函数为 $g(z) = z$，即激活后输出本身，激活函数的导数为1。
- 池化层在前向传播的时候，对输入矩阵进行了压缩，我们需要反向推导出 $\delta^{l-1}$，这个方法与DNN完全不同。
- 卷积层通过张量卷积，或者说若干个矩阵卷积求和得到当前层的输出，而DNN的全连接层是直接进行矩阵乘法而得到当前层的输出。我们需要反向推导出 $\delta^{l-1}$，计算方法与DNN也不同。
- 对于卷积层，由于 $W$ 使用的是卷积运算，那么从 $\delta^l$ 推导出该层的filter的 $W, b$ 方式也不同。

在研究过程中，需要注意的是，由于卷积层可以有多个卷积核，各个卷积核的处理方法是完全相同且独立的，为了简化算法公式的复杂度，我们下面提到卷积核都是卷积层中若干卷积核中的一个。

下面将对上述问题进行逐一分析：

# 已知池化层的 $\delta^l$，推导上一隐藏层的 $\delta^{l-1 }$

在前向传播算法时，池化层一般我们会用MAX或者Average对输入进行池化，池化的区域大小已知。现在我们反过来，要从缩小后的误差 $ \delta^l $，还原前一次较大区域对应的误差。

在反向传播时，我们首先会把 $ \delta^l $ 的所有子矩阵矩阵大小还原成池化之前的大小，然后如果是MAX，则把 $ \delta^l $ 的所有子矩阵的各个池化局域的值放在之前做前向传播算法得到最大值的位置。如果是Average，则把 $ \delta^l $ 的所有子矩阵的各个池化局域的值取平均后放在还原后的子矩阵位置。这个过程一般叫做 $upsample$。

## 示例

假设池化区域为2\*2，步长为2，$\delta^l$ 的第k个子矩阵为：
$$
\delta_k^l = \left(
				\begin{array}{ccc}
					2 & 8 \\
					4 & 6
				\end{array}
			 \right)
$$
我们先将 $ \delta_k^l $ 还原，即变成：
$$
\left(
    \begin{array}{cccc}
    0 & 0 & 0 & 0 \\
    0 & 2 & 8 & 0 \\
    0 & 4 & 6 & 0 \\
    0 & 0 & 0 & 0 \\
    \end{array}
\right)
$$
如果是MAX，假设我们之前在前向传播时记录的最大值位置分别是左上、右下、右上、左下，则转换后的矩阵为：
$$
\left(
    \begin{array}{cccc}
    2 & 0 & 0 & 0 \\
    0 & 0 & 0 & 8 \\
    0 & 4 & 0 & 0 \\
    0 & 0 & 6 & 0 \\
    \end{array}
\right)
$$
如果是Average，转换后的矩阵为：
$$
\left(
    \begin{array}{cccc}
    0.5 & 0.5 & 2 & 2 \\
    0.5 & 0.5 & 2 & 2 \\
    1 & 1 & 1.5 & 1.5 \\
    1 & 1 & 1.5 & 1.5 \\
    \end{array}
\right)
$$
这样我们就得到了上一层 $ \frac {\partial J(W, b)} {\partial a_k^{l-1}} $ ，要得到 $\delta_k^{l-1}$ ：
$$
\delta_k^{l-1} = (\frac {\partial a_k^{l-1}} {\partial z_k^{l-1}})^T \frac {\partial J(W, b)} {\partial a_k^{l-1}} = upsample(\delta_k^l) \odot \sigma'(z_k^{l-1})
$$
其中，$upsample$ 函数完成了池化误差矩阵放大与误差重新分配的逻辑。

对于张量 $\delta^l$ ，我们有：
$$
\delta^{l-1} = upsample(\delta^l) \odot \sigma'(z^{l-1})
$$

## 已知卷积层的 $\delta^l$，推导上一层隐藏层的 $\delta^{l-1}$

在DNN中，我们知道 $\delta^{l-1}$ 和 $\delta^l$ 的递推关系为：
$$
\delta^{l-1} = \frac {\partial J(W, b)} {\partial z^{l-1}} = (\frac {\partial z^l} {\partial z^{l-1}})^T \frac {\partial J(W, b)} {\partial z^l} = (\frac {\partial z^l} {\partial z^{l-1}})^T \delta^l
$$
注意到 $z^l$ 和 $z^{l-1}$ 的关系为：
$$
z^l = a^{l-1}*W^l + b^l = \sigma(z^{l-1})*W^l + b^l
$$
因此我们有：
$$
\delta^{l-1} = (\frac {\partial z^l} {\partial z^{l-1}})^T \delta^l = \delta^l * rot180(W^l) \odot \sigma'(z^{l-1})
$$
这里的式子其实和DNN的类似，区别在于对于含有卷积的式子求导时，卷积核被旋转了180度。即式子中的 $rot180()$，翻转180度的意思是上下翻转一次，接着左右翻转一次。在DNN中这里只是矩阵的转置。那么为什么呢？由于这里都是张量，直接推演参数太多了。我们以一个简单的例子说明为啥这里求导后卷积核要翻转。

假设我们 $l-1$ 层的输出 $a^{l-1}$ 是一个3\*3的矩阵，第 $l$ 层的卷积核 $W^l$ 是一个2\*2矩阵，步幅为1，则输出 $z^l$ 是一个 2\*2的矩阵，这里 $b^l$ 简化为0，则有：
$$
a^{l-1} * W^l = z^l
$$
我们列出 $a, W, z$ 的矩阵表达式如下：
$$
\left(
    \begin{array}{ccc}
    a_{11} & a_{12} & a_{13} \\
    a_{21} & a_{22} & a_{23} \\
    a_{31} & a_{32} & a_{33} \\
    \end{array}
\right) * 
\left(
    \begin{array}{cc}
    w_{11} & w_{12} \\
    w_{21} & w_{22} \\
    \end{array}
\right)
= 
\left(
    \begin{array}{cc}
    z_{11} & z_{12} \\
    z_{21} & z_{22} \\
    \end{array}
\right)
$$
根据卷积得出：
$$
z_{11} = a_{11}w_{11} + a_{12}w_{12} + a_{21}w_{21} + a_{22}w_{22} \\
z_{12} = a_{12}w_{11} + a_{13}w_{12} + a_{22}w_{21} + a_{23}w_{22} \\
z_{21} = a_{21}w_{11} + a_{22}w_{12} + a_{31}w_{21} + a_{32}w_{22} \\
z_{22} = a_{22}w_{11} + a_{23}w_{12} + a_{32}w_{21} + a_{33}w_{22} \\
$$
接着我们模拟反向求导：
$$
\bigtriangledown a^{l-1} = \frac {\partial J(W, b)} {\partial a^{l-1}} = (\frac {\partial z^l} {\partial a^{l-1}})^T \frac {\partial J(W, b)} {\partial z^l} = (\frac {\partial z^l} {\partial a^{l-1}})^T \delta^l
$$
从上式可以看出，对于 $ a^{l-1} $ 的梯度误差 $\bigtriangledown a^{l-1} $ ，等于第 $l$ 层的梯度误差乘以 $\frac {\partial z^l} {\partial a^{l-1}}$ ，而 $\frac {\partial z^l} {\partial a^{l-1}}$ 对应上面的例子中相关联的 $w$ 的值。假设 $z$ 矩阵对应的反向传播误差是 $\delta_{11}, \delta_{12}, \delta_{21}, \delta_{22} $ 组成的2\*2矩阵，则利用上面梯度的式子和4个等式，我们可以分别写出 $\bigtriangledown a^{l-1}$ 的9个标量的梯度。

比如对于 $a_{11}$ 的梯度，由于在4个等式中 $a_{11}$ 只和 $z_{11}$ 有乘积关系，从而我们有：
$$
\bigtriangledown a_{11} = w_{11}\delta_{11}
$$
对于 $a_{12}$ 的梯度，由于在4个等式中 $a_{12}$ 和 $z_{11}, z_{12}$ 有乘积关系，从而我们有：
$$
\bigtriangledown a_{12} = w_{11}\delta_{12} + w_{12}\delta_{11}
$$
同理可得：
$$
\begin{equation}
\begin{aligned}
\bigtriangledown a_{13} &= w_{12}\delta_{12} \\
\bigtriangledown a_{21} &= w_{11}\delta_{21} + w_{21}\delta_{11} \\
\bigtriangledown a_{22} &= w_{11}\delta_{22} + w_{12}\delta_{21} + w_{21}\delta_{12} + w_{22}\delta_{11} \\
\bigtriangledown a_{23} &= w_{12}\delta_{22} + w_{22}\delta_{12} \\
\bigtriangledown a_{31} &= w_{21}\delta_{21} \\
\bigtriangledown a_{32} &= w_{21}\delta_{22} + w_{22}\delta_{21}\\
\bigtriangledown a_{33} &= w_{22}\delta_{22} \\
\end{aligned}
\end{equation}
$$
这上面9个式子其实可以用一个矩阵卷积的形式表示，即：
$$
\left(
    \begin{array}{cccc}
    0 & 0 & 0 & 0 \\
    0 & \delta_{11} & \delta_{12} & 0 \\
    0 & \delta_{21} & \delta_{22} & 0 \\
    0 & 0 & 0 & 0
    \end{array}
\right) * 
\left(
    \begin{array}{cc}
    w_{22} & w_{21} \\
    w_{12} & w_{11} \\
    \end{array}
\right)
= 
\left(
    \begin{array}{cc}
    \bigtriangledown a_{11} & \bigtriangledown a_{12} & \bigtriangledown a_{13} \\
    \bigtriangledown a_{21} & \bigtriangledown a_{22} & \bigtriangledown a_{23} \\
    \bigtriangledown a_{31} & \bigtriangledown a_{32} & \bigtriangledown a_{33} 
    \end{array}
\right)
$$
为了符合梯度计算，我们在误差矩阵周围填充了一圈0，此时我们将卷积核翻转后和反向传播的梯度误差进行卷积，就得到了前一次的梯度误差。这个例子直观的介绍了为什么对含有卷积的式子反向传播时，卷积核要翻转180度的原因。

以上就是卷积层的误差反向传播过程。

## 已知卷积层的 $\delta^l$，推导该层的 $W, b$ 的梯度 

对于全连接层，可以按DNN的反向传播算法求该层 $W, b$ 的梯度，而池化层并没有 $W, b$ ,也不用求 $W, b$ 的梯度。只有卷积层的 $W, b$ 需要求出。

注意到卷积层 $z$ 和 $W, b$ 的关系为：
$$
z^l = a^{l-1} * W^l + b
$$
因此我们有：
$$
\frac {\partial J(W, b)} {\partial W^l} = a^{l-1} * \delta^l
$$
注意到此时卷积核并没有反转，主要是此时是层内的求导，而不是反向传播到上一层的求导。具体过程我们可以分析一下。

这里举一个简化的例子，这里输入是矩阵，不是张量，那么对于第 $l$ 层，某个卷积核矩阵 $W$ 的导数可以表示如下：
$$
\frac {\partial J(W, b)} {\partial W_{pq}^l} = \sum_i \sum_j(\delta_{ij}^l a_{i+p-1, j+q-1}^{l-1})
$$
　那么根据上面的式子，我们有：
$$
\frac {\partial J(W, b)} {\partial W_{11}^l} = a_{11}\delta_{11} + a_{12}\delta_{12} + a_{21}\delta_{21} + a_{22}\delta_{22} \\

\frac {\partial J(W, b)} {\partial W_{12}^l} = a_{12}\delta_{11} + a_{13}\delta_{12} + a_{22}\delta_{21} + a_{23}\delta_{22} \\

\frac {\partial J(W, b)} {\partial W_{13}^l} = a_{13}\delta_{11} + a_{14}\delta_{12} + a_{23}\delta_{21} + a_{24}\delta_{22} \\
...... \\
\frac {\partial J(W, b)} {\partial W_{33}^l} = a_{33}\delta_{11} + a_{34}\delta_{12} + a_{43}\delta_{21} + a_{44}\delta_{22} \\
$$
最终我们可以一共得到9个式子。整理成矩阵形式后可得：
$$
\frac {\partial J(W, b)} {\partial W^l} = 
\left(
    \begin{array}{cccc}
    a_{11} & a_{12} & a_{13} & a_{14} \\
    a_{21} & a_{22} & a_{23} & a_{24} \\
    a_{31} & a_{32} & a_{33} & a_{34} \\
    a_{41} & a_{42} & a_{43} & a_{44}
    \end{array}
\right)
*
\left(
    \begin{array}{cccc}
    \delta_{11} & \delta_{12} \\
    \delta_{21} & \delta_{22}
    \end{array}
\right)
$$
从而可以清楚的看到这次我们为什么没有反转的原因。

而对于 $b$，则稍微有些特殊，因为 $ \delta^l $ 是高维张量，而 $b$ 只是一个向量，不能像DNN那样直接和 $ \delta^l $ 相等。通常的做法是将 $ \delta^l $ 的各个子矩阵的项分别求和，得到一个误差向量，即为 $b$ 的梯度：
$$
\frac {\partial J(W, b)} {\partial b^l} = \sum_{u, v}(\delta^l)_{u, v}
$$

## CNN反向传播算法总结

现在我们总结下CNN的反向传播算法，以最基本的批量梯度下降法为例来描述反向传播算法。

输入：m个图片样本，CNN模型的层数L和所有隐藏层的类型，对于卷积层，要定义卷积核的大小K，卷积核子矩阵的维度F，填充大小P，步幅S。对于池化层，要定义池化区域大小k和池化标准（MAX或Average），对于全连接层，要定义全连接层的激活函数（输出层除外）和各层的神经元个数。梯度学习率 $\alpha$,最大迭代次数MAX与停止迭代阈值 $\epsilon$

输出：CNN模型各隐藏层与输出层的 $W, b$

1. 初始化各隐藏层与输出层的各 $W, b$ 的值为一个随机值。
2. for iter to 1 to MAX:
   1. for i =1 to m：
      1. 将CNN输入 $a^1$ 设置为 $x_i$ 对应的张量
      2.  for l = 2 to L-1，根据下面3种情况进行前向传播算法计算：
         - 如果当前是全连接层：则有 $a^{i, l} = \sigma(z^{i, l}) = \sigma(W^l a^{i, l-1} + b^l)  $
         - 如果当前是卷积层：则有 $a^{i, l} = \sigma(z^{i, l}) = \sigma(W^l * a^{i, l-1} + b^l)  $
         - 如果当前是池化层：则有 $a^{i, l} = pool(a^{i, l-1})$
      3. 对于输出层第L层：$a^{i, L} = softmax(z^{i, L}) = softmax(W^L a^{i, L-1} + b^L)$
      4. 通过损失函数计算输出层的 $\delta^{i, L}$
      5. for l = L-1 to 2, 根据下面3种情况进行进行反向传播算法计算：
         -  如果当前是全连接层：$ \delta^{i, l} = (W^{l+1})^T \delta^{i, l+1} \odot \sigma'(z^{i, l}) $
         - 如果当前是卷积层：$ \delta^{i, l} = \delta^{i, l+1} * rot180(W^{l+1}) \odot \sigma'(z^{i, l}) $
         - 如果当前是池化层：$ \delta^{i, l} = upsample(\delta^{i, l+1}) \odot \sigma'(z^{i, l}) $
   2. for l = 2 to L，根据下面2种情况更新第 $l$ 层的 $W^l, b^l$：
      - 如果当前是全连接层：$W^l = W^l - \alpha \sum_{i=1}^m \delta^{i, l}(a^{i, l-1})^T, b^l = b^l - \alpha \sum_{i=1}^m \delta^{i, l} $
      - 如果当前是卷积层，对于每一个卷积核有：$ W^l = W^l - \alpha \sum_{i=1}^m \delta^{i, l} * a^{i, l-1}, b^l = b^l - \alpha \sum_{i=1}^m \sum_{u, v}(\delta^{i, l})_{u, v} $
   3. 如果所有的 $W, b$ 的变化值都小于停止迭代阈值 $\epsilon$，则跳出迭代循环到步骤3。
3.  输出各隐藏层与输出层的线性关系系数矩阵 $W$ 和偏置量 $b$ 。

___

## 转载
- [卷积神经网络(CNN)反向传播算法](https://www.cnblogs.com/pinard/p/6494810.html?tdsourcetag=s_pcqq_aiomsg)