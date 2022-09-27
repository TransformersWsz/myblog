---
title: wsl2中volume挂载位置的问题
mathjax: true
toc: true
date: 2022-09-27 23:37:32
categories:
- OS
tags:
- Docker
- Volume
---
本人电脑环境：win10 + wsl2(Ubuntu 18.04.6 LTS)

运行命令：`docker inspect testvol`

<!--more-->

```bash
[
    {
        "CreatedAt": "2022-09-27T15:15:09Z",
        "Driver": "local",
        "Labels": {},
        "Mountpoint": "/var/lib/docker/volumes/testvol/_data",
        "Name": "testvol",
        "Options": {},
        "Scope": "local"
    }
]
```
虽然docker给出了挂载的数据位置，但实际上该路径并不存在。之前以为是 `sudo` 权限的问题，还是不能解决。

经过一番摸索后，发现wsl2其实将该路径映射到了windows的路径上。不同的docker版本所映射的路径有所不同：
- Docker Engine v20.10.17：`\\wsl$\docker-desktop-data\data\docker\volumes`

![example](./wsl2中volume挂载位置的问题/example.jpg)
- Docker Engine v19.03: `\\wsl$\docker-desktop-data\version-pack-data\community\docker\volumes\`

___

## 参考
- [Locating data volumes in Docker Desktop (Windows)](https://stackoverflow.com/questions/43181654/locating-data-volumes-in-docker-desktop-windows/64418064#64418064)
- [Docker Desktop for windows原理](https://blog.csdn.net/qqhappy8/article/details/106819429)