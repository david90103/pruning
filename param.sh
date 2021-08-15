#!/bin/bash

REGION=( 4 )
SEARCHER=( 1 )
SAMPLE=( 4 )
C_THRE=( 0.3 0.5 0.7 )
M_THRE=( 0.01 0.02 )

dir=output/params/sem/run2


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
                    python resprune.py  --model "origin/resnet56/model_best.pth.tar"    \
                                        --depth 56                                      \
                                        --save ${dir}/${a}/pruned                   \
                                        --region ${r}                               \
                                        --searcher ${sr}                            \
                                        --sample ${sa}                              \
                                        --mthre ${m}                                \
                                        --cthre ${c}                                \
                                        --algo sem                                  \
                                        >> ${path}
                done
            done
        done
    done
done

echo "Parameter search script done."
