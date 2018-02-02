#!/bin/bash
IMAGENET_PATH=$1
SAVE_PATH=$2

mkdir -p $SAVE_PATH
for dir in `find $IMAGENET_PATH -type d -maxdepth 1 -mindepth 1`; do
   echo $dir
   mkdir -p ${SAVE_PATH}/${dir##*/} 
   for name in ${dir}/*.JPEG; do
      #echo $name
      convert -resize 256x256^ -quality 95 -gravity center -extent 256x256 $name ${SAVE_PATH}/${dir##*/}/${name##*/}
   done
done

