#!/bin/bash

git add .
git commit -m "update"
git push -u origin main
if [[ $? -eq 0 ]]
then
    echo "push to github.com success!"
else
    echo "push to github.com failed!"
fi
