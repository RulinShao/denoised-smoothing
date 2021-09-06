export IMAGENET_DIR=/home/xc150/certify/discrete/smoothing-master


python train_denoiser.py \
  --dataset imagenet \
  --objective stability \
  --noise_sd 0.25 \
  --arch imagenet_dncnn \
  --outdir denoiser/vit_base/sigma_25 \
  --classifier vit_base_patch16_224 \
  --epochs 25 \
  --lr 1e-5 \
  --batch 64
