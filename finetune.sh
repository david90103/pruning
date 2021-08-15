#!/bin/bash

runs=5
dir=output/run45/sem/pruned/vgg16

mkdir -p ${dir}

for r in $(seq 1 $runs)
do
    python main_finetune.py                         \
    --refine ${dir}/run${r}/pruned_best.pth.tar     \
    --save ${dir}/run${r}/                          \
    --arch vgg                                      \
    --depth 16                                      \
    > ${dir}/run${r}/finetune.log

    # python main_finetune.py                         \
    # --refine ${dir}/run${r}/pruned_best.pth.tar     \
    # --save ${dir}/run${r}/                          \
    # --arch resnet                                   \
    # --depth 110                                     \
    # > ${dir}/run${r}/finetune.log
done