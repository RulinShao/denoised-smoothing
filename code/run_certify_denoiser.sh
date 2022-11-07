#!/bin/bash
export IMAGENET_DIR=/l/users/hongyiwa/datasets/ILSVRC2012/
dataset=cifar10

# CUDA_VISIBLE_DEVICES=1 python certify_diffusion_baseline.py \
#   --dataset $dataset \
#   --sigma 0.25 \
#   --outfile certi_diffusion_baseline_$dataset/zero_shot_clip_large/sigma_25 \
#   --skip 10 \
#   --dual_alpha 1.0

CUDA_VISIBLE_DEVICES=0 python certify_clip_diffusion.py \
  --dataset $dataset \
  --sigma 0.25 \
  --outfile certi_clip_diffusion_$dataset/zero_shot_clip_large/sigma_25 \
  --skip 10 \
  --dual_alpha 1.0

# CUDA_VISIBLE_DEVICES=1 python certify_clip.py \
#   --dataset $dataset \
#   --sigma 0.5 \
#   --outfile certi_clip_$dataset/zero_shot_clip_large/sigma_50 \
#   --skip 10 \
#   --dual_alpha 1.0 &

# CUDA_VISIBLE_DEVICES=3 python certify_clip.py \
#   --dataset $dataset \
#   --sigma 1.0 \
#   --outfile certi_clip_$dataset/zero_shot_clip_large/sigma_100 \
#   --skip 10 \
#   --dual_alpha 1.0 &