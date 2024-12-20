#!/bin/bash
DATASET_PATH='/media/ivan/a68c0147-4423-4f62-8e54-388f4ace9ec54/Datasets/SHREC2022/dataset/test'
BIM_TYPE=$(wc -l < $1)

for i in $(find $DATASET_PATH -name '*.txt');
do
    python evaluation.py --file=$i --outf=./output --type=${BIM_TYPE}
done