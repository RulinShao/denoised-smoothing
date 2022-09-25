#!/bin/bash
export IMAGENET_DIR=/home/ubuntu/data/ilsvrc2012/

# # for model in vgg16 deit_tiny_patch16_224 deit_small_patch16_224 deit_small_distilled_patch16_224
# for model in vgg16 deit_tiny_distilled_patch16_224
# do
#   echo $model
#   python certify.py \
#     --dataset imagenet \
#     --base_classifier $model \
#     --sigma 0.25 \
#     --outfile certi_deno/$model/sigma_25 \
#     --skip 20 \
#     --denoiser denoiser/$model/sigma_25/best.pth.tar
# done


CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 python certify_clip.py \
  --dataset imagenet \
  --sigma 0.25 \
  --outfile certi_deno/clip_vit16_b_denoising/sigma_25 \
  --skip 10 \
  --denoiser denoiser/clip_vit16_denoising/sigma_25/best.pth.tar \
  --dual_alpha 0.0 

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 python certify_clip.py \
  --dataset imagenet \
  --sigma 0.5 \
  --outfile certi_deno/clip_vit16_b_denoising/sigma_50 \
  --skip 10 \
  --denoiser denoiser/clip_vit16_denoising/sigma_50/best.pth.tar \
  --dual_alpha 0.0 