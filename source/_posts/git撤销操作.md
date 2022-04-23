---
title: git撤销操作
mathjax: true
date: 2021-05-10 22:16:10
categories:
- VCS
tags:
- Git
---

`git`有如下三种状态：

<!--more-->

{% asset_img git.jpg %}

> - Modified: You have changed the file but have not committed it to your database yet.
> - Staged: You have marked a modified file in its current version to go into your next commit snapshot.
> - Committed: The data is safely stored in your local database.

现在主要讲述下`git`的撤销操作：

## discard changes in working directory

前提：`<file>`已经被添加到暂存区

```plain
git restore <file>
```
## working direction <- index

```plain
git reset HEAD <file>
```
## index <- HEAD

```plain
git reset --soft HEAD^
```
## working direction <- HEAD

```plain
git reset --hard HEAD^
```

