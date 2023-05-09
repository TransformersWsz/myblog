---
title: sh脚本运行过程中修改程序引发的问题
mathjax: true
toc: true
date: 2022-08-12 00:49:21
categories:
- OS
tags:
- Shell
- Bash
- Linux
---
在公司运行shell脚本（暂命名为A.sh）的时候，由于要跑多个应用，所以其依赖的其它shell脚本（暂命名为B.sh）都要经过不同的处理。当A.sh运行的时候（命令还没有走到运行B.sh）：

- 修改A.sh里的任一命令时，会报错
- 修改B.sh里的任一命令时，不会影响A.sh的执行。当执行B.sh的时候，输出的就是更新后的B.sh对应的结果


<!--more-->

在编程语言中，不管是动态语言还是静态语言，程序运行前都会经过编译或者解释生成可执行文件，运行起来后修改源代码，都不会影响程序的正常运行。但shell脚本不同，它只是纯粹文本，有系统一行行读取命令执行。下面举例讨论：

## 例1
`A.sh` 没有依赖其它脚本：
```bash
echo "start running"
sleep 30s
echo "end running"
```
当程序休眠30s的时候，修改 `A.sh` 为：
```bash
echo "start running"
sleep 30s
echo "sleep end"
echo "end running"
```
这个时候程序会报错！
![error](https://img-blog.csdnimg.cn/img_convert/819c68fdc6a719f382725b790d8d002a.webp?x-oss-process=image/format,png)

## 例2
`A.sh` 依赖其它脚本：
```bash
echo "start running"
sleep 30s
sh B.sh
echo "end running"
```

`B.sh` 内容为：
```bash
echo "Here is B"
```

当程序休眠30s的时候，修改 `B.sh` 为：
```bash
echo "Here is B"
echo "Hello B"
```
这个时候程序的输出就会受到影响，结果为：
```bash
start running
Here is B
Hello B
end running
```

## 总结
我们在更改sh脚本的时候，需要极其小心，以免各应用的运行会互相污染。
