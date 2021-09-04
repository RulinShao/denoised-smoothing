export IMAGENET_DIR=/home/xc150/certify/discrete/smoothing-master


python3 train_denoiser.py \
  --dataset imagenet \
  --objective stability \
  --noise_sd 0.25 \
  --arch imagenet_dncnn \
  --outdir denoiser/resnet152/sigma_25 \
  --classifier resnet152 \
  --epochs 25 \
  --lr 1e-5 \
  --batch 64
