export IMAGENET_DIR=/l/users/hongyiwa/datasets/ILSVRC2012/

# train cifar linear head
CUDA_VISIBLE_DEVICES=2 python train_linear_probe.py --gpu 0 --train_clf_head --model_type ViT-B/16 --dataset cifar10 --clf_head_ckpt head_ckpt/cifar10_clip_vit_B16_nn2_clf.pth --batch-size 1024 &
# CUDA_VISIBLE_DEVICES=0 python train_linear_probe.py --gpu 0 --train_clf_head --model_type ViT-L/14 --dataset cifar10 --clf_head_ckpt head_ckpt/cifar10_clip_vit_L14_nn2_clf.pth --batch-size 1024 &
# train imagenet mlp head
CUDA_VISIBLE_DEVICES=3 python train_linear_probe.py --gpu 0 --train_clf_head --model_type ViT-B/16 --dataset imagenet --clf_head_ckpt head_ckpt/imagenet_clip_vit_B16_nn2_clf.pth --batch-size 1024 &
# CUDA_VISIBLE_DEVICES=1 python train_linear_probe.py --gpu 0 --train_clf_head --model_type ViT-L/14 --dataset imagenet --clf_head_ckpt head_ckpt/imagenet_clip_vit_L14_nn2_clf.pth --batch-size 1024 &