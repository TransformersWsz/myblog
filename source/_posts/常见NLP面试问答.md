---
title: 常见NLP面试问答
mathjax: true
date: 2021-03-30 22:58:28
categories:
- Algorithm
tags:
- 面试
---

NLP面试中的经典八股文。

<!--more-->

##  1. HMM vs MEMM vs CRF
#### HMM -> MEMM
HMM模型中存在两个假设：
1. 输出观察值之间严格独立。MEMM解决了HMM输出独立性假设的问题。因为HMM只限定在了观测与状态之间的依赖，而MEMM引入自定义特征函数，不仅可以表达观测之间的依赖，还可表示当前观测与前后多个状态之间的复杂依赖。
2. 状态的转移过程中当前状态只与前一状态有关。但实际上序列标注问题不仅和单个词相关，而且和观察序列的长度，单词的上下文，等等相关。

#### MEMM -> CRF:
- CRF不仅解决了HMM输出独立性假设的问题，还解决了MEMM的标注偏置问题，MEMM容易陷入局部最优是因为只在局部做归一化，而CRF统计了全局概率，在做归一化时考虑了数据在全局的分布，而不是仅仅在局部归一化，这样就解决了MEMM中的标记偏置的问题。使得序列标注的解码变得最优解。
- HMM、MEMM属于有向图，所以考虑了x与y的影响，但没将x当做整体考虑进去（这点问题应该只有HMM）。CRF属于无向图，没有这种依赖性，克服此问题。
___

## 2. 常见的几种优化器

1. SGD

$$
\theta \leftarrow \theta-\eta \nabla_{\theta} J(\theta)
$$
$\eta$ 是学习率，$J(\theta)$ 是损失函数

2. Momentum

$$
\begin{array}{l}
v_{t}=\gamma v_{t-1}+\eta \nabla_{\theta} J(\theta) \\
\theta=\theta-v_{t}
\end{array}
$$

当我们将一个小球从山上滚下来时，没有阻力的话，它的动量会越来越大，但是如果遇到了阻力，速度就会变小。
加入的这一项，可以使得梯度方向不变的维度上速度变快，梯度方向有所改变的维度上的更新速度变慢，这样就可以加快收敛并减小震荡。

**超参数设定值:  一般 γ 取值 0.9 左右。**

3. Nesterov

$$
\begin{array}{l}
v_{t}=\gamma v_{t-1}+\eta \nabla_{\theta} J\left(\theta-\gamma v_{t-1}\right) \\
\theta=\theta-v_{t}
\end{array}
$$

用 $\theta-\gamma v_{t-1}$ 来近似当做参数下一步会变成的值，则在计算梯度时，不是在当前位置，而是未来的位置上。

4. AdaGrad

这个算法就可以对低频的参数做较大的更新，对高频的做较小的更新，也因此，对于稀疏的数据它的表现很好，很好地提高了 SGD 的鲁棒性。
$$
\theta_{t+1, i}=\theta_{t, i}-\frac{\eta}{\sqrt{G_{t, i i}+\epsilon}} \cdot g_{t, i}
$$
其中 $g_{t,i}$ 是 $t$ 时刻参数 $\theta_{i}$ 的梯度，$G_{t, ii}$ (对角矩阵 $G_t$ 的 $(i,i)$ 元素)就是 $t$ 时刻参数 $\theta_i$ 的梯度平方和。

超参数设定值：一般 $\eta$ 选取0.01

5. RMSprop

RMSprop 都是为了解决 Adagrad 学习率急剧下降问题的：
$$
\begin{array}{l}
E\left[g^{2}\right]_{t}=0.9 E\left[g^{2}\right]_{t-1}+0.1 g_{t}^{2} \\
\theta_{t+1}=\theta_{t}-\frac{\eta}{\sqrt{E\left[g^{2}\right]_{t}+\epsilon}} g_{t}
\end{array}
$$
使用的是指数加权平均，旨在消除梯度下降中的摆动，与Momentum的效果一样，某一维度的导数比较大，则指数加权平均就大，某一维度的导数比较小，则其指数加权平均就小，这样就保证了各维度导数都在一个量级，进而减少了摆动。允许使用一个更大的学习率 $\eta$ 。

超参数设定值： $\gamma$ 为 0.9，$\eta$ 为 0.001

6. Adam

相当于 RMSprop + Momentum：
$$
\begin{array}{l}
m_{t}=\beta_{1} m_{t-1}+\left(1-\beta_{1}\right) g_{t} \\
v_{t}=\beta_{2} v_{t-1}+\left(1-\beta_{2}\right) g_{t}^{2}
\end{array}
$$
除了像 momentum 一样保持了过去梯度 $m_t$ 的指数衰减平均值， 也像Adadelta 和 RMSprop 一样存储了过去梯度的平方 $v_t$ 的指数衰减平均值。

如果 $m_t$ 和 $v_t$ 都被初始化为0，那么它们会向0偏置，要做偏差纠正。通过计算偏差校正后的 $m_t$ 和 $v_t$ 来抵消这些偏差：
$$
\hat{m}_t = \frac{m_t} {1-\beta_1^t} \\
\hat{v}_t = \frac{v_t} {1-\beta_2^t}
$$
梯度更新规则：
$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t}+\epsilon} \hat{m}_t
$$
超参数设定值： $\beta_1$ 为 0.9，$\beta_2$ 为0.999，$\epsilon$ 为 $10^{-8}$
___

## 3. Self-Attention添加head数量是否会增加计算复杂度？

不会。self-attention的时间复杂度为$O(n^2 \times d)$，$n$ 为序列长度，$d$ 为维度。假设分成 $h$ 个头，那么张量shape为 $h \times n \times m$ 。其中 $d = h \times m$ 。每个头做self-attention的时间复杂度为 $O(n^2 \times m)$ ，那么 $h$ 个头的总时间复杂度为 $O(h \times n^2 \times m) = O(n^2 \times d)$ 。因此增加头的数量不会导致计算复杂度增加。
___
## 4. L1和L2正则化

### L1正则化
- 优点：输出具有稀疏性，即产生一个稀疏模型，进而可以用于特征选择；一定程度上，L1可以防止过拟合
- 缺点：非稀疏情况下计算效率低

### L2正则化
- 优点：计算效率高（因为存在解析解）；可以防止模型过拟合
- 缺点：非稀疏输出；无特征选择
___
## 5. 方差与偏差
> 偏差：用所有可能的训练数据集训练出的所有模型的输出的平均值与真实模型的输出值之间的差异。
> 方差：不同的训练数据集训练出的模型输出值之间的差异。

- 欠拟合：高偏差，低方差
- 过拟合：高偏差，高方差
___
## 6. 正负样例分布不均衡解决办法
1. 过采样与欠采样：
	- 过抽样：通过增加分类中少数类样本的数量来实现样本均衡
	- 欠抽样：通过减少分类中多数类样本的数量来实现样本均衡
2. 通过正负样本的惩罚权重解决样本不均衡：对于分类中不同样本数量的类别分别赋予不同的权重，一般是小样本量类别权重高，大样本量类别权重低。
___
## 7. 词汇表太大，softmax计算如何优化？
Hierarchical Softmax根据单词出现的频率来构建一颗霍夫曼树。树的叶子结点代表一个单词，在每一个非叶子节点处都需要作一次二分类，走左边的概率和走右边的概率，这里用逻辑回归的公式表示：
- 正类别：$\sigma\left(X_{i} \theta\right)=\frac{1}{1+e^{-x_{i} \theta}}$
- 负类别：$1-\sigma\left(X_{i} \theta\right)$
每个词都会有一条路径，根据训练样本的特征向量 $X_i$ 预测目标label词 $Y_i$ 的概率为：
$$
P\left(Y_{i} \mid X_{i}\right)=\prod_{j=2}^{l} P\left(d_{j} \mid X_{i}, \theta_{j-1}\right) \\
P\left(d_{j} \mid X_{i}, \theta_{j-1}\right)=\left\{\begin{array}{ll}
\sigma\left(X_{i} \theta\right), & \text { if } \mathrm{d_j}=1 \\
1-\sigma\left(X_{i} \theta\right), & \text { if } \mathrm{d_j}=0
\end{array}\right.
$$
详细见：
- [层次softmax函数（hierarchical softmax）](https://www.cnblogs.com/eniac1946/p/8818892.html)
- [Hierarchical Softmax（层次Softmax）](https://zhuanlan.zhihu.com/p/56139075)
## 8. LSTM如何解决梯度弥散或爆炸？
LSTM的介绍见：[LSTM：RNN最常用的变体](https://zhuanlan.zhihu.com/p/44124492)
梯度问题见：
- [LSTM如何解决梯度消失或爆炸的？](https://www.cnblogs.com/bonelee/p/10475453.html)
- [LSTM如何来避免梯度弥散和梯度爆炸？](https://www.zhihu.com/question/34878706)
___
## 9. 简述EM算法的流程
输入：观察数据 $x=\left(x^{(1)}, x^{(2)}, \ldots x^{(m)}\right),$ 联合分布 $p(x, z ; \theta),$ 条件分布 $p(z \mid x ; \theta),$ 最大迭代次数 $J$
1. 随机初始化模型参数 $\theta$ 的初值 $\theta^{0}$
2. $for \quad j \quad from \quad 1 \quad to \quad j$:
a) E步。计算联合分布的条件概率期望：
$$
\begin{array}{c}
Q_{i}\left(z^{(i)}\right)=P\left(z^{(i)} \mid x^{(i)}, \theta^{j}\right) \\
L\left(\theta, \theta^{j}\right)=\sum_{i=1}^{m} \sum_{z^{(i)}} Q_{i}\left(z^{(i)}\right) \log P\left(x^{(i)}, z^{(i)} ; \theta\right)
\end{array}
$$
b) M步。极大化 $L\left(\theta, \theta^{j}\right),$ 得到 $\theta^{j+1}$ :
$$
\theta^{j+1}=\underset{\theta}{\arg \max } L\left(\theta, \theta^{j}\right)
$$
c) 如果 $\theta^{j+1}$ 收敛, 则算法结束。否则继续回到步骤 a) 进行E步迭代
输出：模型参数 $\theta$ 。
### 具体示例可见：
- [人人都懂EM算法](https://zhuanlan.zhihu.com/p/36331115)
- [如何通俗理解EM算法](https://blog.csdn.net/v_JULY_v/article/details/81708386)
- [EM算法原理总结](https://www.cnblogs.com/pinard/p/6912636.html)
___
## 10. LR、SVM、决策树的对比
### LR
#### 优点
1. 实现简单高效
2. 对观测样本概率输出
#### 缺点
1. 特征空间太大时表现不太好
2. 对于非线性特征须要作特征变换
3. 需要额外添加正则项
### SVM
#### 优点
1. 能够处理高维特征 
2. 自带正则项
3. 使用核函数轻松应对非线性特征空间
4. 分类面不依赖于全部数据
#### 缺点
1. 核函数选择较难
2. 样本量非常大，核函数映射维度非常高时，计算量过大
3. 对缺失数据敏感

### 决策树
#### 优点
1. 决策过程直观
2. 可以处理非线性特征
#### 缺点
1. 容易过拟合
2. 无法输出概率，只能输出分类结果
___
## 11. SVM常用核函数
1. 线性核函数
2. 多项式核函数
3. 高斯核函数
4. sigmoid核函数

详细见：[svm常用核函数](https://blog.csdn.net/batuwuhanpei/article/details/52354822)
___
## 12. k-means与EM联系与区别
> 两者都是无监督学习。

k-means可以看成是两阶段的：
- 第一阶段，确定每一个样本所属的聚类，在这个过程中，聚类的中心保持不变。可以看作EM的E步。
- 第二阶段，确定聚类中心，在这个过程中，每一个样本所属的类别保持不变。可以看作EM的M步。

EM算法和K-Means算法的迭代过程比较类似，不同的是K-Means算法中每次对参数的更新是硬猜测，而EM中每次对参数的更新是软猜测；相同的是，两个算法都可能得到局部最优解，采用不同的初始参数迭代会有利于得到全局最优解。

详细见：
- [机器学习笔记11: K-Means算法和EM算法](https://www.jianshu.com/p/2c42c567e893)
- [k-Means与EM之间的关系](https://www.cnblogs.com/youyouzaLearn/p/9471409.html)
___
## 13. Xavier原理
> 为了使得网络中信息更好的流动，每一层输出的方差应该尽量相等。

先贴结论：
$$
w \sim U\left[-\frac{\sqrt{6}}{\sqrt{n_{i n}+n_{\text {out }}}}, \frac{\sqrt{6}}{\sqrt{n_{\text {in }}+n_{\text {out }}}}\right]
$$

具体的公式推导见：
- [深度前馈网络与Xavier初始化原理
](https://zhuanlan.zhihu.com/p/27919794)
- [深度学习之参数初始化（一）——Xavier初始化](https://blog.csdn.net/VictoriaW/article/details/73000632)
- [一文搞懂深度网络初始化（Xavier and Kaiming initialization）](https://www.jianshu.com/p/f2d800388d1c)
___
## 14. 目标检测综述
### Two-stage方法
#### R-CNN
1. 通过Selective Search（SS）方法筛选出一些备选的区域框（Region proposal）；
2. CNN提取特征，SVM分类；
3. 分类完成后，对bbox进行回归，修正bbox中的坐标的值，得到更精确的bbox。

#### SPP-net
- R-CNN中，每个区域都要过一次CNN 提取特征。而SPP-net中，一张图片只需要过一次CNN，特征提取是针对整张图进行的，候选区域的框定以及特征向量化是在CNN的feature map层面进行的。
- 提出自适应池化的方法，它分别对输入的feature map（可以由不定尺寸的输入图像进CNN得到，也可由region proposal 框定后进CNN 得到）进行多个尺度（实际上就是改变pooling 的size 和stride）的池化，分别得到特征，并进行向量化后拼接起来。无需像R-CNN一样对所有的Region proposal进行缩放得到相同的大小。

#### Fast RCNN
- 提出了ROI pooling 的结构，实际上就是一种特殊的SPP（相当于SPP 的金字塔层数设置为了1，即只计算一次池化）。
- 将最终的SVM分类去掉了，直接做成了端到端的一个网络结构。对这个网络进行多任务训练，即分类和回归，得到物体类别和bbox的位置。

#### Faster R-CNN
提出RPN网络：利用一个与检测器共享部分权重的RPN 网络来直接对图片生成候选框，然后基于RPN 得到的候选框进行分类和位置回归：
> 定义anchor box 的尺寸（scale）和比例（aspect ratio）。按上图，预先定义了k个anchor box。在实际的RPN网络实现中，共采用了3个不同的scale（大中小）和3种不同的比例（宽中窄）。然后通过组合，得到了9个anchor box，即 $k=9$ 。在训练RPN的过程中，对于每个feature map上的像素点，都生成 $k$ 个anchor box 的预测。由于预测需要有两个输出用来分类（前景/背景），以及4个用来定位 $(x, y, w, h)$ ，所以RPN的分类层生成的是 $2k$ 维度的向量，RPN的回归层生成的是 $4k$ 维度的向量。

### One-stage方法
#### YOLO v1
YOLO的过程如下：首先，将整个图像分成 $S \times S$ 的小格子（cell），对于每个格子，分别预测B 个bbox，以及C个类别的条件概率（注意是条件概率，即已经确定有目标的情况下，该目标属于哪个类别的概率，因此不需要对每个bbox分别预测类别，每个格子只预测一个概率向量即可）。每个bbox都有5个变量，分别是四个描述位置坐标的值，以及一个objectness，即是否有目标（相当于RPN 网络里的那个前景/背景预测）。这样一来，每个格子需要输出 $5B+C$ 维度的向量，因此，CNN最终的输出的tensor的形态为 $S \times S \times (5B + C)$ 。

YOLO的训练过程如下：首先，对于每个GT bbox，找到它的中心位置，该中心位置所在的cell负责该物体的预测。因此，对于该cell 中的输出，其objectness应该尽可能的增加，同时其位置坐标尽可能拟合GTbbox（注意，由于每个cell可以输出多个备选的bbox，因此这里需要选择和GT最相近的那个预测的bbox进行调优）。另外，根据其实际的类别，对类别概率向量进行优化，使其输出真实的类别。对于不负责任何类别的那些cell 的预测值，不必进行优化。

#### SSD
SSD 也是一种one-stage的直接检测的模型。它相比起YOLO v1主要的改进点在于两个方面：
1. 利用了先验框（Prior Box）的方法，预先给定scale 和aspect ratio，实际上就是之前Faster R-CNN 中的anchor box的概念。
2. 多尺度（multi-scale）预测，即对CNN输出的后面的多个不同尺度的feature map 都进行预测。

#### YOLO v2
1. 对所有卷积层增加了BN层。
2. 用高分辨率的图片fine-tune 网络10个epoch。
3. 通过k-means进行聚类，得到 $k$ 个手工选择的先验框（Prior anchor box）。这里的聚类用到的距离函数为 $1 - IoU$ ，这个距离函数可以很直接地反映出IoU 的情况。
4. 直接预测位置坐标。之前的坐标回归实际上回归的不是坐标点，而是需要对预测结果做一个变换才能得到坐标点，即 $x = tx \times wa − xa$ （纵坐标同理），其中 $tx$ 为预测的直接结果。从该变换的形式可以看出，对于坐标点的预测不仅和直接预测位置结果相关，还和预测的宽和高也相关。因此，这样的预测方式可以使得任何anchor box可以出现在图像中的任意位置，导致模型可能不稳定。在YOLO v2 中，中心点预测结果为相对于该cell的角点的坐标（0-1 之间）。
5. 多尺度训练（随机选择一个缩放尺度）、跳连层（paththrough layer）将前面的fine-grained特征直接拼接到后面的feature map 中。

#### FPN
通过将所有scale 的feature map 进行打通和结合，兼顾了速度和准确率。

FPN的block 结构分为两个部分：一个自顶向下通路（top-down pathway），另一个是侧边通路（lateral pathway）。所谓自顶向下通路，具体指的是上一个小尺寸的feature map（语义更高层）做2倍上采样，并连接到下一层。而侧边通路则指的是下面的feature map（高分辨率低语义）先利用一个1x1 的卷积核进行通道压缩，然后和上面下来的采样后结果进行合并。合并方式为逐元素相加（element-wise addition）。合并之后的结果在通过一个3x3的卷积核进行处理，得到该scale下的feature map。

#### RetinaNet
RetinaNet 的最大的贡献不在于网络结构，而是在于提出了一个one-stage 检测的重要的问题，及其对应的解决方案。这个问题就是one-stage 为何比two-stage 的准确率低，两者的区别在哪里？解决方案就是平衡正负样本+平衡难易样本的focal loss。

#### Mask R-CNN
本模型将实例分割（instance segmentation）与目标检测（object detection）两个任务相结合，并在两个任务上都达到了SOTA。

整个过程的pipeline 如下：首先，输入图片，根据RoI进行RoIAlign操作，得到该区域内的特征，然后将该特征feature map 进行逐点sigmoid（pixel-wise sigmoid），用于产生mask。另外，还有两个支路用于分类和回归。

#### YOLO v3
YOLO v3 是针对YOLO模型的又一次改进版本，是一个incremental improvement，并无太大创新，基本都是一些调优和trick。主要包括以下几个方面。

1. 用单类别的binary logistic 作为分类方式，代替全类别的softmax（和mask R-CNN 的mask 生成方式类似）。这样的好处在于可以处理有些数据集中有目标重叠的情况。

2. YOLO v3采用了FPN网络做预测，并且沿用了k-means聚类选择先验框，v3中选择了9个prior box，并选择了三个尺度。

3. backbone做了改进，从darknet-19变成了darknet-53，darknet-53除了3x3和1x1的交替以外，还加入了residual方法，因此层数得到扩展。

### 参考自
- [从R-CNN到YOLO，2020 图像目标检测算法综述](https://mp.weixin.qq.com/s/Hh5EioN_pVnstfHcR777VQ)
___
## 15. 过拟合的原因
- 训练集的数量和模型的复杂度不匹配，比如训练集太小或者模型太复杂
- 训练集和测试集分布不一致
- 训练集的噪声样本太多，导致模型只学习到了噪声特征，反而忽略了真实的输入输出关系