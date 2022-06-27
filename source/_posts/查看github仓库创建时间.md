---
title: 查看github仓库创建时间
mathjax: true
toc: true
date: 2022-03-17 20:13:17
categories:
- 软件工具
tags:
- github
---
github是没有直接的图形化界面来显示仓库的最早创建时间的，我们可以通过调用api的形式来查看，格式如下：


```bash
https://api.github.com/repos/{username}/{reponame}
```
<!--more-->

## 最佳实践

- 确定 `username` 和 `reponame` ：
![1](https://cdn.jsdelivr.net/gh/TransformersWsz/image_hosting@master/1.5wtegw7m1ow0.webp)
- 在终端中输入：

```bash
curl -k  https://api.github.com/repos/bojone/bert4keras | jq . | grep created_at
```

- 结果如下：
![2](https://cdn.jsdelivr.net/gh/TransformersWsz/image_hosting@master/case.2krf83ywy1g0.webp)
___

## 参考

- [一键查看GitHub仓库的创建日期](https://blog.csdn.net/Jack_lzx/article/details/117480746)
- [bojone/bert4keras: keras implement of transformers for humans](https://github.com/bojone/bert4keras)