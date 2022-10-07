---
title: Pandas入门
mathjax: true
toc: true
date: 2022-08-19 00:48:26
categories:
- Software Engineering
tags:
- Python
- Pandas
---
日常工作中经常需要数据分析，以前都是python脚本读取文件然后统计，十分麻烦。尝试了下Pandas，真香！

<!--more-->

```python
!cat data
```

    123	600
    23	67
    563	456
    345	345



```python
import pandas as pd
```

## 读取txt文件


```python
df = pd.read_table("./data", sep="\t", header=None, names=["A", "B"])    # 添加自定义列名：A, B
print(df)
```

         A    B
    0  123  600
    1   23   67
    2  563  456
    3  345  345


## 添加新的列


```python
df["C"] = df["B"] / df["A"]
print(df)
print(df[["A", "C"]])
```

         A    B         C
    0  123  600  4.878049
    1   23   67  2.913043
    2  563  456  0.809947
    3  345  345  1.000000
         A         C
    0  123  4.878049
    1   23  2.913043
    2  563  0.809947
    3  345  1.000000


## 条件过滤

- 单条件


```python
t = df[df["C"]>2.5]
print(t)
```

         A    B         C
    0  123  600  4.878049
    1   23   67  2.913043


- 多条件


```python
t = df[(df["A"]>100) & (df["C"]<3)]
print(t)
```

         A    B         C
    2  563  456  0.809947
    3  345  345  1.000000


## 保存到文件


```python
df.to_csv("./final.txt", sep="\t", index=False, header=None)
```


```python
!cat final.txt
```

    123	600	4.878048780487805
    23	67	2.9130434782608696
    563	456	0.8099467140319716
    345	345	1.0


## 列过滤并求和

```python
!cat stu.csv
```

    name,age
    Tome,12
    Jack,NULL
    Node,13
    Node,NULL

```python
df = pd.read_csv("./stu.csv")
a= df[df["age"].notna()]["age"].sum()
print(a)
```

    25.0

___

## 参考
- [pandas里面按条件筛选](https://zhuanlan.zhihu.com/p/87334662)
