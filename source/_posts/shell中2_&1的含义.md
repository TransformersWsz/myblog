---
title: shell中2>&1的含义
mathjax: true
toc: true
date: 2021-07-16 00:50:25
categories:
- OS
tags:
- Linux
- Shell
---

在做实验的过程中，经常会看到shell脚本里存在 `2>&1` 的指令组合，有点懵逼，在此记录一下。

<!--more-->


## `0`  |  `1`  |  `2`
这三个数字是linux文件描述符。
|   数字   |  含义    |
| ---- | ---- |
|   0   |    stdin   |
|   1   |    stdout  |
|   2   |    stderr  |

`1` 和 `2` 都是输出到屏幕设备上。

我们平时使用的：
```bash
echo "Hello World" > test.log
```

等价于：

```bash
echo "Hello World" 1> test.log
```
注意 `1>` 是连在一块的。如果分开，那么写入文件的就是 <font color="red">Hello World 1</font> 。

## `2>&1`
> 将标准错误输出重定向到标准输出。

### 示例
```python
import sys
sys.stdout.write("this is stdout\n")
sys.stderr.write("this is stderr\n")
```

```bash
python test.py > test.log 2>&1
```
表示将标准错误输出重定向到标准输出，标准输出重定向到 `test.log` 文件中。

在程序执行过程中，我们希望输出保存到文件中，如果有错误的信息也希保存到文件中，那么就使用上面的命令。


- `python test.py > test.log 2>1` : 那么标准输出将重定向到 `test.log` ，而错误将输出到<font color="red">名字为 `1` 的文件中</font>。这里的 `&` 可以理解为 `1` 的引用
- `python test.py > test.log 2 >&1` : 那么标准输出将重定向到 `test.log` ，而错误将输出到屏幕上
- `python test.py > test.log 2>& 1` : 等价于 `python test.py > test.log 2>&1`
- `python test.py 1> test.log 2> test.log` : 会存在如下两个问题：
    - `stdout` 会覆盖 `stderr`
    - `test.log` 会被打开两次，IO效率低下

### `2>&1` 为什么放在末尾？
`python test.py > test.log 2>&1` 从左往右来看 `stdin` 定向到 `test.log`，然后 `stderr` 定向到 `stdin`，等于说 `stderr` 也输入到了 `test.log` 中。

如果放在中间：`python test.py 2>&1 >test.log` ，`stderr` 定向到 `stdin`，但此时 `stdin` 指向的是屏幕，所以 `stderr` 会输出到屏幕。执行到 `>test.log` 的时候，`stdin` 定向到 `test.log`。所以 `test.log` 文件里只有 `this is stdout`。

### 简写

```bash
python test.py > test.log 2>&1
```

可以简写成：

```bash
python test.py >& test.log
```
___
## 参考
- [Linux shell中2>&1的含义解释](https://blog.csdn.net/zhaominpro/article/details/82630528)