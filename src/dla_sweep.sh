#!/bin/bash

for i in 0.01 0.001 0.0001 0.00001
do
  ./scripts/train_adaptive_dla34.sh $i
done
