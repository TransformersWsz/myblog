---
title: python2.7安装tensorflow1.x
mathjax: true
toc: true
date: 2024-12-09 01:58:44
updated: 2024-12-09 01:58:44
categories:
- Software Engineering
tags:
- python2.7
- tensorflow1.x
---

当前tensorflow官方已不再提供1.x版本的pip安装，尝试了网上多种解决方案后，最简单的就是换源。

<!--more-->

编辑`~/.pip/pip.conf`，将pip源换成清华源：
```yaml
[global]
timeout = 6000
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn
```