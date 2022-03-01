# Regularizing Brain Age Prediction via Gated Knowledge Distillation

## Description
Brain age prediction with  Gated Knowledge Distillation regularization method using Pytorch. The method was trained on 4 datasets:

IXI: http://brain-development.org/

OASIS-3: https://www.oasis-brains.org/

ADNI: https://ida.loni.usc.edu/

1000-FCP: http://www.nitrc.org/projects/fcon_1000

## Setup
All MRI images need to be preprocessed and z-score normalization.

The preprocess pipeline code is in preparation

We recommend the normalization using: https://github.com/jcreinhold/intensity-normalization

### requirements:
>apex  
>transformations  
>pytorch  
>numpy  
>logging  
>nibabel  
>sklearn

Install apex using source code: https://github.com/NVIDIA/apex  
Others can be installed by pip or conda  

Run :

Train Teacher model:
```
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
                        --nproc_per_node=1 \
                        --master_port 51321 \
                        run.py \
                        -b 64 \
                        --epochs 300 \
                        --lr 0.0003 \
                        -p 50  \
                        --arch sfcn \
                        --data /path/to/data \
                        --T 10
```  
Train Student model by adding teacher path (-t)
```shell
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
                        --nproc_per_node=1 \
                        --master_port 51321 \
                        run.py \
                        -b 64 \
                        --epochs 300 \
                        --lr 0.0003 \
                        -p 50  \
                        --arch sfcn \
                        --data /path/to/data \
                        --T 10 \
                        -t /path/to/model
```

Cite:  
>{@article{yang2021regularizing,  
  title={Regularizing Brain Age Prediction via Gated   Knowledge Distillation},  
  author={Yang, Yanwu and Xutao, Guo and Ye, Chenfei and Xiang, Yang and Ma, Ting},  
  year={2021}  
}