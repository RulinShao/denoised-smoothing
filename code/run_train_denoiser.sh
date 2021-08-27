export IMAGENET_DIR=~/dataset/ilsvrc2012


python3 train_denoiser.py \
  --dataset imagenet \
  --objective stability \
  --noise_sd 0.25 \
  --arch imagenet_dncnn \
  --outdir denoiser/resnet34/sigma_25 \
  --classifier resnet34 \
  --epochs 25 \
  --lr 1e-5 \
  --batch 64
