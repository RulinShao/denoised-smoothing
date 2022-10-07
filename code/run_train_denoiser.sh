#!/bin/bash
export IMAGENET_DIR=/home/ubuntu/data/ilsvrc2012/

# for model in t2t_vit_14 t2t_vit_t_14 t2t_vit_24 t2t_vit_t_24
# for model in ViT-B/16
# do
#   python train_denoiser_clip.py \
#     --dataset imagenet \
#     --objective stability \
#     --noise_sd 0.25 \
#     --arch imagenet_dncnn \
#     --outdir denoiser/clip_vit16/sigma_25 \
#     --classifier ViT-B/16 \
#     --epochs 25 \
#     --lr 1e-5 \
#     --batch 64
# done


CUDA_VISIBLE_DEVICES=0,1,2,3 python train_denoiser_clip.py \
    --dataset imagenet \
    --objective clip_feat_denoising \
    --noise_sd 0.5 \
    --arch imagenet_dncnn \
    --outdir denoiser/clip_vit16/clip_feat_denoising/sigma_50 \
    --classifier ViT-B/16 \
    --clf_head_ckpt /home/ubuntu/RobustCLIP/head_ckpt/imagenet/vit16/clip_vit16_nn2_clf.pth \
    --epochs 25 \
    --lr 1e-5 \
    --batch 64 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python train_denoiser_clip.py \
    --dataset imagenet \
    --objective clip_feat_denoising \
    --noise_sd 1.0 \
    --arch imagenet_dncnn \
    --outdir denoiser/clip_vit16/clip_feat_denoising/sigma_100 \
    --classifier ViT-B/16 \
    --clf_head_ckpt /home/ubuntu/RobustCLIP/head_ckpt/imagenet/vit16/clip_vit16_nn2_clf.pth \
    --epochs 25 \
    --lr 1e-5 \
    --batch 64 &