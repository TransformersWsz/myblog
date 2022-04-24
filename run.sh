#!/bin/sh

# add\commit\push

git add .
git commit -m ${1}
git push -u origin main

cp correct/hexo-renderer-kramed/renderer.js node_modules/hexo-renderer-kramed/lib/
cp correct/kramed/inline.js node_modules/kramed/lib/rules/ 
cp correct/hexo-renderer-mathjax/mathjax.html node_modules/hexo-renderer-mathjax