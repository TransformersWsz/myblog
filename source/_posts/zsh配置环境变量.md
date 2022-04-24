---
title: zsh配置环境变量
mathjax: true
toc: true
date: 2022-03-10 16:08:36
categories:
- OS
tags:
- zsh
---

MacOS现在默认的shell为zsh了，这里以配置node环境变量为例：

<!--more-->

1. 打开 `~/.zshrc`
2. 输入如下内容：
```shell
NODE_ENV=~/opt/node/bin
export PATH=$NODE_ENV:$PATH
```
3. `source ~/.zshrc`

## 注意
如果只是输入：`export NODE_ENV=~/opt/node/bin` ，那么终端还是不能识别 `node` 命令，只能输出 `echo $NODE_ENV` ，必须要把 `NODE_ENV` 加入到 `PATH` 中。