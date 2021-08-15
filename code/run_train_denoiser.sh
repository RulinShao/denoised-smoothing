export IMAGENET_DIR=~/dataset/ilsvrc2012

python3 train_denoiser.py --dataset imagenet --arch imagenet_dncnn --outdir denoiser/vit_small/ --classifier vit_small_patch16_224 --epochs 25 --lr 1e-5 --batch_size 64