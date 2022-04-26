---
title: 利用Github Action来自动化部署Hexo博客
mathjax: true
toc: true
date: 2022-04-24 23:19:04
categories:
- 软件工具
tags:
- hexo
- CI
---
这两天尝试了使用Github Action来自动化部署博客，踩了一些坑，在此记录一下。

<!--more-->

## 新建仓库
- 存放博客源文章的仓库（Source Repo），命名随意
- 存放编译后生成的静态文件的仓库（Page Repo），命名`username.github.io`

## 配置部署密钥
利用 `ssh-keygen` 来生成公钥和私钥：
- 私钥放于Source仓库的 `Settings -> Secrets -> Actions` ，新建一个secret，命名为 `HEXO_DEPLOY_PRI`：
{% asset_img pri.png %}

- 公钥放于Page仓库的 `Settings -> Deploy Keys` ，新建一个deploy key，命名随意：
{% asset_img pub.jpg %}


## 编写Action
整个Source仓库的结构如下所示：
{% asset_img tree.jpg %}
只需要保留源文件就行了，其它的依赖交给Action来安装。


在 `.github/workflows` 新建 `deploy.yml` 文件，内容如下：

```yaml
# workflow name
name: Hexo Blog CI

# master branch on push, auto run
on: 
  push:
    branches:
      - main
      
jobs:
  build: 
    runs-on: ubuntu-latest 
        
    steps:
    # check it to your workflow can access it
    # from: https://github.com/actions/checkout
    - name: Checkout Repository master branch
      uses: actions/checkout@master 
      
    # from: https://github.com/actions/setup-node  
    - name: Setup Node.js 16.x 
      uses: actions/setup-node@master
      with:
        node-version: "16.14.0"
    
    - name: Cache node modules
      uses: actions/cache@v1    # 缓存node_modules，避免每次跑action都要重新下载
      id: cache
      with:
        path: node_modules
        key: ${{ runner.os }}-node-${{ hashFiles('**/package-lock.json') }}
        restore-keys: |
          ${{ runner.os }}-node-
    
    - name: Setup Hexo Dependencies
      run: |
        npm install hexo-cli -g
        npm install
    
    - name: update mathjax    # kramed引擎有点问题，将其部分文件替换掉
      run: |
        cp correct/hexo-renderer-kramed/renderer.js node_modules/hexo-renderer-kramed/lib/
        cp correct/kramed/inline.js node_modules/kramed/lib/rules/
        cp correct/hexo-renderer-mathjax/mathjax.html node_modules/hexo-renderer-mathjax

    - name: Setup Deploy Private Key
      env:
        HEXO_DEPLOY_PRIVATE_KEY: ${{ secrets.HEXO_DEPLOY_PRI }}    # 这个就是Source仓库的私钥
      run: |
        mkdir -p ~/.ssh/
        echo "$HEXO_DEPLOY_PRIVATE_KEY" > ~/.ssh/id_rsa 
        chmod 600 ~/.ssh/id_rsa
        ssh-keyscan github.com >> ~/.ssh/known_hosts
        
    - name: Setup Git Infomation
      run: | 
        git config --global user.name "TransformersWsz"
        git config --global user.email "3287124026@qq.com"
    - name: Deploy Hexo 
      run: |
        hexo clean
        hexo generate 
        hexo deploy
```

### 相关字段说明
- `use`：引用现有的第三方的action，这样就无需自己写流程了
- `run`：运行命令，用法跟linux一致

## FAQ
### 1. 自己使用的主题未生效？
- 原因：由于主题是 `git clone` 下来的，主题目录下生成了 `.git` 目录，导致和 hexo根目录下 `.git` 冲突了，commit时没有把主题push上去导致的。
- 解决办法： 删掉主题下的 `.git` 文件夹，重新提交，目的是把主题文件夹提交上去（删掉 `.git` 文件夹后git commit依然没有提交上，需要把主题文件夹剪切出来后 `git add && git commit && git push` 后，再把主题文件夹拷贝回来，再 `git add && git commit && git push` 就可以提交成功了）


___

## 参考
- [Github action自动部署Hexo Next](https://blog.csdn.net/liuhp123/article/details/114040409)
