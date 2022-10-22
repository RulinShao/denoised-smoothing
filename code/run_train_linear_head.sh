export IMAGENET_DIR=/home/ubuntu/data/ilsvrc2012/

# train cifar linear head
python train_linear_probe.py --gpu 7 --train_clf_head --model_type ViT-L/14 --dataset cifar10 --clf_head_ckpt head_ckpt/cifar10_clip_vit_L14_nn2_clf.pth &
# train imagenet mlp head
python train_linear_probe.py --gpu 6 --train_clf_head --model_type ViT-L/14 --dataset imagenet --clf_head_ckpt head_ckpt/imagenet_clip_vit_L14_nn2_clf.pth &