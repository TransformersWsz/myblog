#!/bin/bash

git add .
git commit -m "update"
git push -u origin main

if [[ $? -eq 0 ]]
then
    echo "push success!"
else
    echo "push failed!"
fi