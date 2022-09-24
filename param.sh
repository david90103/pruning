#!/bin/bash

REGION=( 2 )
SEARCHER=( 2 )
SAMPLE=( 8 )
C_THRE=( 0.7 )
M_THRE=( 0 )

dir=output/params/thesis/sem/run2


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
                    python vggprune.py  --model "origin/cifar10/vgg16/model_best.pth.tar" \
                                        --depth 16                                  \
                                        --save ${dir}/${a}/pruned                   \
                                        --region ${r}                               \
                                        --searcher ${sr}                            \
                                        --sample ${sa}                              \
                                        --mthre ${m}                                \
                                        --cthre ${c}                                \
                                        --iter 10000000000000                       \
                                        --algo sem                                  \
                                        >> ${path}
                done
            done
        done
    done
done

echo "Parameter search script done."
