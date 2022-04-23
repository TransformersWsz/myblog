#!/bin/sh

# add\commit\push

git add .
git commit -m ${1}
git push -u origin main
