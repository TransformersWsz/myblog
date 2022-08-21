---
title: Linux sort命令
mathjax: true
toc: true
date: 2022-08-22 00:28:43
categories:
- OS
tags:
- Linux
- sort
---
`sort` 命令用于对字符串排序，在日常的脚本处理中非常有用，用法也很简单。

<!--more-->

有数据文件如下：

```bash
cat fruit.txt
```

    banana;30;5.5
    apple;10;2.5
    pear;90;2.3
    orange;20;3.4

三列信息为水果名称、销售数量、单价。现要求以单价来降序输出这些水果信息：

```bash
sort -t ";" -k 3 -n -r fruit.txt
```

    banana;30;5.5
    orange;20;3.4
    apple;10;2.5
    pear;90;2.3

具体的参数说明：
- `-t`：列分隔符
- `-k`: 取第3列作为排序键
- `-n`：根据数值大小排序，默认是字符ASCII大小排序
- `-r`：逆序输出

___

## 参考
- [linux sort 命令详解](https://www.cnblogs.com/51linux/archive/2012/05/23/2515299.html)
- [Linux sort 命令](https://www.runoob.com/linux/linux-comm-sort.html)