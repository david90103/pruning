#### 以搜尋經濟學演算法修剪卷積神經網路濾波器

GECCO'22 Paper: https://dl.acm.org/doi/abs/10.1145/3520304.3528935


1. 預訓練完整模型
```
python main.py --save origin/cifar10/resnet110  \
               --arch resnet                    \
               --depth 110                      \
               --dataset cifar10
```

2. 使用 SE 進行 CNN 濾波器修剪
```
python resprune.py  --model origin/cifar10/resnet110/model_best.pth.tar      \
                    --save output/sem/pruned/resnet110/run1                  \
                    --algo sem                                               \
                    --depth 110                                              \
                    --dataset cifar10
```

3. 微調修剪後模型
```
python main_finetune.py --refine output/sem/pruned/resnet110/run1/pruned_best.pth.tar   \
                        --dataset cifar10                                               \
                        --arch resnet                                                   \
                        --depth 110                                                     \
                        --save finetuned
```

#### 比較對象 Github 連結
- L1-norm: https://github.com/Eric-mingjie/rethinking-network-pruning
- FPGM: https://github.com/he-y/filter-pruning-geometric-median
- SFP: https://github.com/he-y/soft-filter-pruning
- HRank: https://github.com/lmbxmu/HRank
- GAL: https://github.com/ShaohuiLin/GAL
