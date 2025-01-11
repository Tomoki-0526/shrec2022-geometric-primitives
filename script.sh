#!/bin/bash
DATASET_PATH='/home/szj/SHREC2022/dataset/test/pointCloud'
OUTPUT_DIR='/home/szj/SHREC2022/results/methods/M8/prediction_results'

for i in $(find $DATASET_PATH -name '*.txt');
do
    python evaluation.py --file=$i --outf=${OUTPUT_DIR}
done