---
title: Linux引号
mathjax: true
toc: true
date: 2022-08-08 23:47:06
- OS
tags:
- Shell
- Linux
---
Linux的引号分为单引号、双引号、反引号三种。

<!--more-->

## 单引号
被单引号包括的字符串被看作是普通字符串，不会对特殊字符进行转义：
![single](./Linux引号/single.jpg)

## 双引号
被双引号包括的字符会进行转义：
![double](./Linux引号/double.jpg)

## 反引号
如果需要执行是shell命令的字符串，则使用反引号：
![reverse](./Linux引号/reverse.jpg)

___

## 参考
- [Shell（Bash）单引号、双引号和反引号用法详解](http://c.biancheng.net/view/951.html)