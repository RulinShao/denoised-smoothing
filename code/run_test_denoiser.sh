
export IMAGENET_DIR=/home/xc150/certify/discrete/smoothing-master

python3 test_denoiser.py \
  --dataset imagenet \
  --noise_sd 0.25 \
  --denoiser denoiser/resnet18/sigma_25/best.pth.tar \
  --clf resnet18 \
  --outdir test_deno/resnet18/sigma_25 \
  --batch 128 \
  --gpu 0,1,2,3