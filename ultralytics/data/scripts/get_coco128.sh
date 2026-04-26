#!/bin/bash










d='../datasets'
url=https://github.com/ultralytics/assets/releases/download/v0.0.0/
f='coco128.zip'
echo 'Downloading' $url$f ' ...'
curl -L $url$f -o $f -

wait
