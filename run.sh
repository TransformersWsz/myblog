#!/bin/sh

# echo ${1}

cd source/_posts/
dir=${1}
file=${1}.md

mkdir "${dir}"
touch "${file}"

echo ${dir} " directory has been create"
echo ${file} " markdown has been create"