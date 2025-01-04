#!/bin/bash
DATASET_PATH='/home/szj/SHREC2022/dataset/test/pointCloud'

for i in $(find $DATASET_PATH -name '*.txt');
do
    python evaluation.py --file=$i --outf=/home/szj/SHREC2022/results/methods/M7/prediction_results
done