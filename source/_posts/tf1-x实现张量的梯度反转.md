---
title: tf1.x实现张量的梯度反转
mathjax: true
toc: true
date: 2024-10-17 23:37:02
updated: 2024-10-17 23:37:02
categories:
- Machine Learning
tags:
- Gradient Reversal
- TensorFlow
---

tensorflow实现梯度反转的方法有两种：

<!--more-->

## 利用`@tf.custom_gradient`重写梯度函数
```python
import tensorflow as tf

# 自定义反转梯度的操作
@tf.custom_gradient
def gradient_reverse(x):
    def grad(dy):
        return -dy  # 反转梯度
    return x, grad

# 定义变量 w 和 x
w = tf.Variable(2.0)  # 假设 w 的初始值为 2.0
x = tf.Variable(3.0)  # 假设 x 的初始值为 3.0

# 定义运算 y = w * x
y = w * x
y_reversed = gradient_reverse(y)

# 求 y 关于 x 的梯度
grad_x = tf.gradients(y, x)
grad_x_reversed = tf.gradients(y_reversed, x)

# 初始化变量
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    # 计算 y 的值和 x 的梯度
    y_value, grad_x_value, grad_x_reversed_value = sess.run([y, grad_x, grad_x_reversed])
    print("y: ", y_value)
    print("x gradient: ", grad_x_value)
    print("x gradient reversed: ", grad_x_reversed_value)
```

## 巧用`stop_gradient`函数
以实现[DANN](https://transformerswsz.github.io/2024/09/24/DANN-GRL/)为例，特征提取器记作`F`，域分类器记作`D`，那么`F`梯度反转的实现如下：
```python
feat = F(x)
loss = -D(F(x)) + 2*D(stop_gradient(F(x)))
```
- 在前向传播的过程中，`stop_gradient`不起作用，那么`loss = D(stop_gradient(F(x)))`
- 在反向传播的过程中，`stop_gradient`起作用，那么`2*stop_gradient(F(x)`梯度为0，梯度计算就是`2D-D-F=D-F`，就实现了`F`的梯度反转

___

## 参考
- [在TensorFlow中自定义梯度的两种方法 ](https://www.cnblogs.com/FesianXu/p/13283799.html)