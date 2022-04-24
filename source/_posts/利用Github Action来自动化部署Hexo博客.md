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
      uses: actions/cache@v1
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
    
    - name: update mathjax
      run: |
        cp correct/hexo-renderer-kramed/renderer.js node_modules/hexo-renderer-kramed/lib/
        cp correct/kramed/inline.js node_modules/kramed/lib/rules/
        cp correct/hexo-renderer-mathjax/mathjax.html node_modules/hexo-renderer-mathjax

    - name: Setup Deploy Private Key
      env:
        HEXO_DEPLOY_PRIVATE_KEY: ${{ secrets.HEXO_DEPLOY_PRI }}
      run: |
        mkdir -p ~/.ssh/
        echo "$HEXO_DEPLOY_PRIVATE_KEY" > ~/.ssh/id_rsa 
        chmod 600 ~/.ssh/id_rsa
        ssh-keyscan github.com >> ~/.ssh/known_hosts
        
    - name: Setup Git Infomation
      run: | 
        git config --global user.name "username"
        git config --global user.email "email"
    - name: Deploy Hexo 
      run: |
        hexo clean
        hexo generate 
        hexo deploy
```

___

## 参考
- [Github action自动部署Hexo Next](https://blog.csdn.net/liuhp123/article/details/114040409)
