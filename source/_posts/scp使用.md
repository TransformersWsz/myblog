---
title: scp使用
mathjax: true
toc: true
date: 2021-11-09 22:18:56
updated: 2021-11-09 22:18:56
categories:
- OS
tags:
- Linux
- bash
---

有的时候本地下载好的文件需要上传到服务器上去，但是需要借助第三方软件，显得非常繁琐。因此就用了一下 `scp` 命令：

```bash
scp local_file username@ip:remote_folder
```
