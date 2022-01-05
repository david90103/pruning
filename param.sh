#!/bin/bash

REGION=( 4 )
SEARCHER=( 1 2 )
SAMPLE=( 1 2 4 8 )
C_THRE=( 0.5 )
M_THRE=( 0.1 )

dir=output/params/sem/run3


mkdir -p ${dir}

for r in "${REGION[@]}"
do
    for sr in "${SEARCHER[@]}"
    do
        for sa in "${SAMPLE[@]}"
        do
            for c in "${C_THRE[@]}"
            do
                for m in "${M_THRE[@]}"
                do
                    path=${dir}/sem_r${r}_se${sr}_sa${sa}_c${c}_m${m}.out
                    date > ${path}

                    mkdir -p ${dir}/${a}
                    python resprune.py  --model "origin/cifar10/resnet56/model_best.pth.tar"    \
                                        --depth 56                                      \
                                        --save ${dir}/${a}/pruned                   \
                                        --region ${r}                               \
                                        --searcher ${sr}                            \
                                        --sample ${sa}                              \
                                        --mthre ${m}                                \
                                        --cthre ${c}                                \
                                        --iter 300                                  \
                                        --algo sem                                  \
                                        >> ${path}
                done
            done
        done
    done
done

echo "Parameter search script done."