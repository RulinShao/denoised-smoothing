export IMAGENET_DIR=~/dataset/ilsvrc2012

python3 certify.py \
  --dataset imagenet \
  --base_classifier resnet18 \
  --sigma 0.25 \
  --outfile certi_deno/resnet18 \
  --skip 20 \
  --denoiser denoiser/resnet18/best.pth.tar \
