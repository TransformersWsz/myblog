---
title: 解决wsl2参考的对象类型不支持尝试的操作
mathjax: true
toc: true
date: 2022-09-09 00:52:32
updated: 2022-09-09 00:52:32
categories:
- OS
tags:
- WSL
- 代理
---
最近windows的代理软件出现了问题，导致winsock出现问题，连锁反应就是wsl也用不了了。

<!--more-->

解决方法就是从winsock中排除wsl：

```powershell
Windows Registry Editor Version 5.00
 
[HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Services\WinSock2\Parameters\AppId_Catalog\0408F7A3]
"AppFullPath"="C:\\Windows\\System32\\wsl.exe"
"PermittedLspCategories"=dword:80000000
```

新建文本文档，复制上述代码，后缀修改为reg并双击运行，问题解决。

___

## 转载
- [一劳永逸，wsl2出现“参考的对象类型不支持尝试的操作”的解决办法](https://blog.csdn.net/marin1993/article/details/119841299)