
export IMAGENET_DIR=~/dataset/ilsvrc2012

python3 test_denoiser.py \
  --dataset imagenet \
  --noise_sd 0.25 \
  --denoiser denoiser/resnet18/best.pth.tar \
  --clf vit_small_patch16_224 \
  --outdir test_deno/resnet18/ \
  --batch 128 \
  --gpu 0,1,2,3