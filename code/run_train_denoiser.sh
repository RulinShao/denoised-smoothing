#!/bin/bash
export IMAGENET_DIR=/home/ubuntu/data/ilsvrc2012/

for model in convnext_small convnext_tiny 
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
    --batch 128
done