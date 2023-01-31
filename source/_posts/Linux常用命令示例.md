---
title: Linux常用命令示例
mathjax: true
toc: true
date: 2022-08-07 03:20:10
categories:
- OS
tags:
- Shell
- Bash
- Linux
---
记录一下Linux常用命令的使用示例：

<!--more-->

## du
- 显示某目录下各文件夹大小
```bash
du -h –-max-depth=1 ./
```

## awk

`awk` 高阶使用较为复杂，这里记录一些例子：

### 提取出文件路径中以特定字符串开头的字段
```bash
echo "/user/wl_0/sdf/wl_1/gf" | awk -v FS="/" '{ for (i = 1; i <= NF; ++i) {if ( $i ~ /^wl/ ) print $i} }'
```
结果：
```bash
wl_0
wl_1
```

### 显示一行有多少列
```bash
head -n 1 filename | awk -F '\t' '{print NF}'
```

### 对某列求和
```bash
# 第一列
awk '{sum += $1};END {print sum}' filename
```

### 关于 `BEGIN` 和 `END` 使用
[如何使用 awk 的特殊模式 BEGIN 与 END](https://www.linuxprobe.com/awk-begin-end.html)

## find
- 查找指定目录及子目录下特定文件名的位置：`find ./ -name db.json`
  - 模糊匹配：`find ./ -name "*.json"`

## shuf
- 随机抽取文件的10行
```bash
shuf -n10 filename
```