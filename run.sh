#!/bin/bash

echo "start replacing cdn links with raw github links..."
# 设置要遍历的文件夹路径
folder="./source/_posts"
for file in "$folder"/*; do
    if [ -f "$file" ]; then
        # 如果是文件，则使用 sed 命令替换文件内容
        sed -i 's|https://cdn.jsdelivr.net/gh/TransformersWsz/picx-images-hosting@master|https://raw.githubusercontent.com/TransformersWsz/picx-images-hosting/master|g' "$file"
        # echo "Replaced 'https://cdn.jsdelivr.net/gh/TransformersWsz/picx-images-hosting@master' with 'https://raw.githubusercontent.com/TransformersWsz/picx-images-hosting/master' in file: $file"
    fi
done
echo "end replacing cdn links with raw github links..."

git add .
git commit -m "update"
git push -u origin main
echo "----------------------------------------------------------------------------"
if [[ $? -eq 0 ]]
then
    echo -e "\033[32m\033]8;;https://github.com/TransformersWsz/myblog\apush to https://github.com/TransformersWsz/myblog success~\U0001F680\033]8;;\a\033[0m"

else
    echo -e "\033[31m\033]8;;https://github.com/TransformersWsz/myblog\apush to https://github.com/TransformersWsz/myblog failed~\U0001F605\033]8;;\a\033[0m"
fi
echo "----------------------------------------------------------------------------"