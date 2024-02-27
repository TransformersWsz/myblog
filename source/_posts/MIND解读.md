---
title: MIND解读
mathjax: true
toc: true
date: 2024-02-28 02:29:04
categories:
- 搜广推
tags:
- 召回
- 胶囊网络
---

这篇paper的核心是胶囊网络，该网络采用了动态路由算法自动对用户历史行为序列进行聚类，提取出多个兴趣向量，代表用户的不同兴趣。当用户再有新的交互时，通过胶囊网络，还能实时的改变用户的兴趣表示向量，做到在召回阶段的实时个性化。

<!--more-->

## 前置知识-胶囊网络
本质上就是个聚类网络，将输入的多个向量聚类输出多个向量：

![capsule](https://img-blog.csdnimg.cn/c189e1258de64e42b576884844e718a4.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA57-75rua55qE5bCPQOW8ug==,size_1,color_FFFFFF,t_70,g_se,x_16#pic_center)

算法迭代流程：

![algo](https://img-blog.csdnimg.cn/82746b6ff8ac47fab6a89788d8d50f9e.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA57-75rua55qE5bCPQOW8ug==,size_1,color_FFFFFF,t_70,g_se,x_16#pic_center)

## MIND模型
![MIND](https://img-blog.csdnimg.cn/33b251f8dcb242ad82b2ed0313f6df73.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA57-75rua55qE5bCPQOW8ug==,size_2,color_FFFFFF,t_70,g_se,x_16#pic_center)

#### 训练

用户向量（item向量对用户多个兴趣向量的weighted sum）：

![user vec](https://raw.githubusercontent.com/TransformersWsz/picx-images-hosting/master/image.2h81jn319s.webp)

loss：

![loss](https://raw.githubusercontent.com/TransformersWsz/picx-images-hosting/master/image.3raypyjygs.webp)
由于$D$数量太大，团队采用了负采样技术

#### 在线服务
同时利用MIND模型产出的多个兴趣向量进行ann检索召回，最终排序得到topK个商品

## 疑问
1. 如何确定兴趣胶囊的数量？

团队用了一种启发式方式自适应调整聚类中心的数量：
$$
K_u^{\prime}=\max \left(1, \min \left(K, \log _2\left(\left|\mathcal{I}_u\right|\right)\right)\right)
$$
该超参对实验影响多大，团队并没有在这上面进行深入实验

2. 既然胶囊网络是为了聚类，为什么不直接使用k-means方法？
3. 论文说当用户有新的交互时，通过胶囊网络，还能实时的改变用户的兴趣表示向量，做到在召回阶段的实时个性化。但如果是用户兴趣发生变化了呢？比如之前有两个兴趣胶囊（体育、旅游），现在用户多了个兴趣（数码），那就要新增一个兴趣胶囊，等于模型要重新训练？

___

## 参考

- [fun-rec/MIND](https://github.com/datawhalechina/fun-rec/blob/master/docs/ch02/ch2.1/ch2.1.4/MIND.md)