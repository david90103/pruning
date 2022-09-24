#!/bin/bash

runs=5
dir=output/thesis_svhn/gwo/pruned/

# output/run1/sem/pruned/vgg19/run1/pruned_best.pth.tar

mkdir -p ${dir}

for r in $(seq 1 $runs)
do
    folder=${dir}/vgg16/run${r}
    python main_finetune.py                         \
    --refine ${folder}/pruned_best.pth.tar     	    \
    --save ${folder}/                               \
    --arch vgg                                      \
    --depth 16                                      \
    --epochs 160				    \
    --weight-decay 0.005                            \
    --batch-size 256				    \
    --lr 0.01 					    \
    --dataset svhn \
    > ${folder}/best_finetune.log

    #folder=${dir}/resnet56/run${r}
    #python main_finetune.py                         \
    #--refine ${folder}/pruned_best.pth.tar          \
    #--save ${folder}/                               \
    #--arch resnet                                   \
    #--depth 56                                      \
    #--epochs 200				    \
    #--weight-decay 0.005                            \
    #--batch-size 256				    \
    #--lr 0.01 					    \
    #> ${folder}/best_finetune_200.log

    #folder=${dir}/resnet110/run${r}  
    #python main_finetune.py                         \
    #--refine ${folder}/pruned_best.pth.tar          \
    #--save ${folder}/                               \
    #--arch resnet                                   \
    #--depth 110                                     \
    #--epochs 200				    \
    #--weight-decay 0.005                            \
    #--batch-size 256				    \
    #--lr 0.01 					    \
    #> ${folder}/best_finetune_200.log
done
