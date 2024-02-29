---
title: keep主题从3.x升级到4.x后GitHub Actions自动部署后文章更新时间异常的问题
mathjax: true
toc: true
date: 2024-02-29 10:11:29
categories:
- tools
tags:
- Hexo
- CI
- theme-keep
---
keep主题4.x新增了很多功能配置，在升级的过程中遇到了一些问题，在此记录一下：

<!--more-->

## GitHub Actions自动部署后文章更新时间异常
在首页展示的所有的文章更新时间都是一样的：

![case](https://raw.githubusercontent.com/TransformersWsz/picx-images-hosting/master/image.9kfx1lyvok.webp)

这是因为自动化工作流每次都需要先克隆Hexo博客项目源码，才能进行后续的构建生成和部署等操作。但在Hexo博客中，如果没有在文章的`Front-Matter`设置`updated`，Hexo会默认使用文件的最后修改时间作为文章的更新时间，这就是为什么会出现自动部署后所有文章更新时间都一致的真正原因。