#!/bin/bash

git add .
git commit -m "update"
git push -u origin main
if [[ $? -ne 0 ]]
then
    echo "push to github.com failed!"
else
    echo "push to github.com success!"
fi
