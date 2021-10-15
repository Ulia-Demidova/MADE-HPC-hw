#!/bin/bash

array=( a b c d e f g h i j k )

for i in {1..${#array[@]}..1}
  do
    echo ${array[$i]}
done
