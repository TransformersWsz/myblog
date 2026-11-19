---
title: excel根据百分比绘制数据条
mathjax: true
toc: true
date: 2026-11-20 01:06:45
updated: 2026-11-20 01:06:45
categories:
- 随笔
tags:
- Excel
- 数据条
---
我们可以用数据条让百分比数据看起来更加直观。下面记录操作步骤：

1. 数据准备：工号为第一列，第二列展示每个员工的完工率
![1](https://i-blog.csdnimg.cn/direct/a88ac9e21d3141f2bec36e456f37f6f0.png)
2. 将小数调整成百分比形式
![2](https://i-blog.csdnimg.cn/direct/a4a42730aa024ed39c3ca4e61b24bb67.png)
3. 调用 `条件格式 > 数据条`
![3.1](https://i-blog.csdnimg.cn/direct/aaf6e4081d77451da683076d4fa0a81c.png)
![3.2](https://i-blog.csdnimg.cn/direct/2fc3c90f7daa4e02b1e9c2b151985387.png)
4. 校准基线：由于数据条默认以最大值为满格（上图的245%），需要调整`1.0`为基线

*步骤1：管理规则*
![4.1](https://i-blog.csdnimg.cn/direct/da4811a5a17c4282838eed8d76d9427b.png)
*步骤2：编辑规则*
![4.2](https://i-blog.csdnimg.cn/direct/e8c57df186484c7d962f131c61729c16.png)
*步骤3：编辑最大值*
![4.3](https://i-blog.csdnimg.cn/direct/3828c6950b324e328b905a25363757dc.png)
5. 效果展示
![5](https://i-blog.csdnimg.cn/direct/b1bffa9c043846458f5401e78e8dbb28.png)

