export IMAGENET_DIR=/home/xc150/certify/discrete/smoothing-master

python certify.py \
  --dataset imagenet \
  --base_classifier resnet152 \
  --sigma 0.25 \
  --outfile certi_deno/resnet152/sigma_25 \
  --skip 20 \
  --denoiser denoiser/resnet152/best.pth.tar \
