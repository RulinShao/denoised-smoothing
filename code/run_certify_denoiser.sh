export IMAGENET_DIR=/home/xc150/certify/discrete/smoothing-master

python3 certify.py \
  --dataset imagenet \
  --base_classifier resnet18 \
  --sigma 0.25 \
  --outfile certi_deno/resnet18/sigma_25 \
  --skip 20 \
  --denoiser denoiser/resnet18/best.pth.tar \
