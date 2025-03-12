---
title: git合并多个commit
mathjax: true
toc: true
date: 2022-10-14 01:12:07
updated: 2022-10-14 01:12:07
categories:
- VCS
tags:
- Git
---
有时候为了开发一个新的功能，进行了多次commit，从而导致整个git提交历史显得很冗余。在此记录一下如何合并多个commit：

<!--more-->

我们想把提交历史 `A->B->C->D` 合并成 `A->B->C&D`：


![his](https://github.com/TransformersWsz/picx-images-hosting/raw/master/image.491ffzcfhf.webp)

1. 回到基线B：`git rebase -i 36a1ccf`，然后会看到下图：

![rebase](https://github.com/TransformersWsz/picx-images-hosting/raw/master/image.5tr6fhzx45.webp)

2. `pick` 表示选择这条commit，`squash` 表示将该commit合并到上一个commit，这里我们选择`pick C, squash D`
3. 编辑完上述信息后，退出保存会弹出如下界面。这是因为两个commit合并后会生成一个新的commit，所以要填写message：

![update](https://github.com/TransformersWsz/picx-images-hosting/raw/master/image.45u3xei2e.webp)

![msg](https://github.com/TransformersWsz/picx-images-hosting/raw/master/image.6f0u1t4eb0.webp)
4. 编辑完上述信息后，退出保存。这时候再查看log信息，就发现生效了：
![msg](https://github.com/TransformersWsz/picx-images-hosting/raw/master/image.102bjdq85c.webp)

备注：许多人在用ubuntu系统，默认编辑器是nano，这里可以切换成vim，方便操作：`echo export EDITOR=/usr/bin/vim >> ~/.zshrc && source ~/.zshrc`

___

## 参考

- [Git 多次提交合并成一次提交](https://kunzhao.org/docs/tutorial/git/merge-multiple-commit/)
- [修改ubuntu默认编辑器为vim](https://blog.csdn.net/zhezhebie/article/details/82382984)