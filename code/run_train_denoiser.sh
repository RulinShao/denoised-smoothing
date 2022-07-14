#!/bin/bash
export IMAGENET_DIR=/home/ubuntu/data/ilsvrc2012/

for model in deit_tiny_patch16_224 swin_tiny_patch4_window7_224 vgg16 deit_tiny_distilled_patch16_224 vit_tiny_patch16_22 deit_small_patch16_224 deit_small_distilled_patch16_224 swin_small_patch4_window7_224
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
    --batch 64
done