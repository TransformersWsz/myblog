---
title: awk使用示例
mathjax: true
toc: true
date: 2022-08-07 03:20:10
categories:
- OS
tags:
- Shell
- Bash
---
`awk` 高阶使用较为复杂，这里记录一些例子：

<!--more-->

## 提取出文件路径中以特定字符串开头的字段
```bash
echo "/user/wl_0/sdf/wl_1/gf" | awk -v FS="/" '{ for (i = 1; i <= NF; ++i) {if ( $i ~ /^wl/ ) print $i} }'
```
结果：
```bash
wl_0
wl_1
```