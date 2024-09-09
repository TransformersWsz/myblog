---
title: M1 Mac安装Homebrew
mathjax: true
toc: true
date: 2024-09-10 01:41:11
updated: 2024-09-10 01:41:11
categories:
- tools
tags:
- Homebrew
- ARM
---
Homebrew对ARM芯片的Mac支持不友好，这里切换到国内镜像网站安装，速度快且稳定，没有乱七八糟的报错：

```bash
/bin/bash -c "$(curl -fsSL https://gitee.com/ineo6/homebrew-install/raw/master/install.sh)"
```

___

## 参考
- [Homebrew 中文网](https://brew.idayer.com/)