#!/bin/bash
export IMAGENET_DIR=/home/ubuntu/data/ilsvrc2012/

# for model in t2t_vit_14 t2t_vit_t_14 t2t_vit_24 t2t_vit_t_24
for model in seresnet50
do
  echo $model
  python train_denoiser.py \
    --dataset imagenet \
    --objective stability \
    --noise_sd 0.25 \
    --arch imagenet_dncnn \
    --outdir denoiser/$model/sigma_25 \
    --classifier $model \
    --epochs 25 \
    --lr 1e-5 \
    --batch 16
done