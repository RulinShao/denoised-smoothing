#!bin/bash

export IMAGENET_DIR=~/dataset/ilsvrc2012

if [ $1 = 'train' ]
then
  python3 train_denoiser.py \
    --dataset imagenet \
    --objective stability \
    --noise_sd 0.25 \
    --arch imagenet_dncnn \
    --outdir denoiser/vit_small/ \
    --classifier vit_small_patch16_224 \
    --epochs 25 \
    --lr 1e-5 \
    --batch 64
elif [ $1 = 'test']
then
  python3 test_denoiser.py \
  --dataset imagenet \
  --noise_sd 0.25 \
  --denoiser denoiser/vit_small/best.pth.tar \
  --clf vit_small_patch16_224 \
  --outdir test_deno/vit_small/ \
  --batch 128 \
  --gpu 0,1,2,3
fi

