#!/bin/bash
DATASET_PATH='/home/szj/shrec2022-geometric-primitives/data'
BIM_TYPE=$1

for i in $(find $DATASET_PATH -name '*.txt');
do
    python evaluation.py --file=$i --outf=./output --type=${BIM_TYPE}
done