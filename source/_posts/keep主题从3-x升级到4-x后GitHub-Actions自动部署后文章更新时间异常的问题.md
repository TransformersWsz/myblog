---
title: keep主题从3.x升级到4.x后GitHub Actions自动部署后文章更新时间异常的问题
mathjax: true
toc: true
date: 2024-02-29 19:11:29
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

#### 解决方法
1. 在博客的`Front-Matter`添加`updated`字段：
```markdown
---
title: TDM检索技术讲解
mathjax: true
toc: true
date: 2024-02-27 01:46:38
updated: 2024-02-27 01:46:38
categories:
- 搜广推
tags:
- 召回
- Tree-based Model
---
```

2. 使用主题作者提供的Github Action工作流
```yaml
name: Hexo Deploy GitHub Pages
on:
  push:
    branches:
      - main
jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout
      uses: actions/checkout@v3
      with:
        fetch-depth: 0

    - name: Build and Deploy
      uses: theme-keep/hexo-deploy-github-pages-action@master
      env:
        # Your GitHub Token
        PERSONAL_TOKEN: ${{ secrets.MYBLOG_TOKEN }}

        # The repository the action should deploy to
        PUBLISH_REPOSITORY: TransformersWsz/TransformersWsz.github.io
        BRANCH: main
```

- `PERSONAL_TOKEN`：github token，这样源码仓库`TransformersWsz/myblog`可以写静态文件仓库`TransformersWsz/TransformersWsz.github.io`，获取步骤如下：
    1. 创建[Personal Access Token](https://github.com/settings/tokens)
    2. 在[源码仓库](https://github.com/TransformersWsz/myblog/settings/secrets/actions)添加token

![add_token](https://raw.githubusercontent.com/TransformersWsz/picx-images-hosting/master/image.1aoqdgo4t7.webp)

{% note danger %}
注意添加`BRANCH: main`，否则它会默认推送到静态仓库的[`gh-pages`](https://github.com/theme-keep/hexo-deploy-github-pages-action/blob/e4c3d8b5150fe5bcc2b643cbd056cd034355442b/entrypoint.sh#L16)分支
{% endnote %}

## 参考
- [如何解决 GitHub Actions 自动部署后文章更新时间异常的问题](https://keep.xpoet.cn/2023/11/%E5%A6%82%E4%BD%95%E8%A7%A3%E5%86%B3-GitHub-Actions-%E8%87%AA%E5%8A%A8%E9%83%A8%E7%BD%B2%E5%90%8E%E6%96%87%E7%AB%A0%E6%9B%B4%E6%96%B0%E6%97%B6%E9%97%B4%E5%BC%82%E5%B8%B8%E7%9A%84%E9%97%AE%E9%A2%98/)
- [You must provide the action with either a Personal Access Token or the GitHub Token secret in order to deploy](https://stackoverflow.com/questions/59261555/you-must-provide-the-action-with-either-a-personal-access-token-or-the-github-to)
- [hexo-deploy-github-pages-action](https://github.com/theme-keep/hexo-deploy-github-pages-action)