---
title: 'Linux条件判断X的作用'
mathjax: true
toc: true
date: 2022-08-04 00:59:21
categories:
- OS
tags:
- Linux
---
在shell脚本中经常遇到这样的条件判断：
```bash
if [[ "X"${var} == "X" ]]
then
    echo "null"
fi
```
主要是用来判断变量 `var` 是否为空。

如果不加 `"X"`，判断 `var` 是否等于某一个值，比如 `"0"` ，一旦出现 `var` 为空或者未设置，那么条件表达式就为：
```bash
if [[  == "0" ]]
```
语法错误。加上 `"X"`后就可以避免此错误。

___

## 参考

- [shell if [ “x${var}" == “x” ]中x的作用](https://blog.csdn.net/readnap/article/details/105047518)