#!/bin/bash

#" de" "ga" "random" "gwo" "pso" "sem" "se"
algorithms=( "sem" "pso" "de" "gwo" "ga" "random")
runs=5
dir=output/run1

mkdir -p ${dir}

for run in $(seq 1 $runs)
do
        for a in "${algorithms[@]}"
        do      
                mkdir -p ${dir}/${a}
                cp search.sh ${dir}/${a}

                # python vggprune.py  --model "origin/cifar10/vgg16/model_best.pth.tar"   \
                #                     --save ${dir}/${a}/pruned/vgg16/run${run}           \
                #                     --algo ${a}                                         \
                #                     --depth 16                                          \
                #                     >> "${dir}/${a}/vgg16_${a}_run_${run}.out"

                python vggprune.py  --model "origin/cifar10/vgg19/model_best.pth.tar"  \
                                    --save ${dir}/${a}/pruned/vgg19/run${run}           \
                                    --algo ${a}                                         \
                                    --depth 19                                          \
                                    >> "${dir}/${a}/vgg19_${a}_run_${run}.out"

                # python resprune.py      --model "origin/cifar10/resnet56/model_best.pth.tar"    \
                #                         --save ${dir}/${a}/pruned/resnet56/run${run}    \
                #                         --algo ${a}                                     \
                #                         --depth 56                                      \
                #                         --iter 100              \
                #                         --region 4              \
                #                         --searcher 1            \
                #                         --sample 4              \
                #                         >> "${dir}/${a}/resnet56_${a}_run_${run}.out"

                # python resprune.py      --model "origin/resnet110/model_best.pth.tar"   \
                #                         --save ${dir}/${a}/pruned/resnet110/run${run}   \
                #                         --algo ${a}                                     \
                #                         --depth 110                                     \
                #                         --iter 100              \
                #                         --region 4              \
                #                         --searcher 1            \
                #                         --sample 4              \
                #                         >> "${dir}/${a}/resnet110_${a}_run_${run}.out"
        done
done

echo "Search script done."
