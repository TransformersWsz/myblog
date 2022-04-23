#!/bin/sh

# add\commit\push

git add .
git commit -m ${1}
git push -u origin main

# 第n个参数（大于10的时候必须使用花括号）
${n}
