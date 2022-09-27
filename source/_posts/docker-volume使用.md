---
title: docker volume使用
mathjax: true
toc: true
date: 2022-09-27 23:04:44
categories:
- OS
tags:
- Docker
- Volume
---
`volume` 是docker官方提供的一种高级的持久化数据的方法，它比 `mount` 有如下优点：

<!--more-->

- 更容易备份和迁移
- 可以使用docker命令行或者api来管理volume
- 兼容linux和windows容器
- 可以在多个容器间共享
- 可加密存储在远程机器或云端
- 新volume可以被容器预填充
- volume性能比mount更高

![volume](https://docs.docker.com/storage/images/types-of-mounts-volume.png)

下面是volume的一些基础命令：

## 创建volume
`docker volume create testvol`
## 查看volume信息
`docker volume inspect testvol`
## 删除volume
`docker volume rm myvol`
