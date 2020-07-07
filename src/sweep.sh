#!/bin/bash

for i in 0.0001 0.00001
do
  for j in 0.5 0.6 0.7 0.8 0.9
  do
    echo $i $j
    ./scripts/train_adaptive_stream_pretrained.sh $i $j
  done
done
