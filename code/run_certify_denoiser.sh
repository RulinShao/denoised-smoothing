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


python certify_clip.py \
  --dataset imagenet \
  --sigma 0.25 \
  --outfile certi_deno/clip_vit16/sigma_25 \
  --skip 20 \
  --denoiser denoiser/clip_vit16/sigma_25/best.pth.tar