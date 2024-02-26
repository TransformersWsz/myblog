#!/bin/bash

git add .
git commit -m "update"
git push -u origin main
if [[ $? -eq 0 ]]
then
    echo -e "\033[32mpush to github.com success!\033[0m"

else
    echo -e "\033[31mpush to github.com failed!\033[0m"

fi
