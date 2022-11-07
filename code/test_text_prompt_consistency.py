"""
compute the statistics of text prompting class with noisy image inputs
check if the predictions are consistent

export IMAGENET_DIR=/l/users/hongyiwa/datasets/ILSVRC2012/
"""
from architectures import get_architecture, IMAGENET_CLASSIFIERS
from core import Smooth, SmoothOptimizeAlpha
from datasets import get_dataset, DATASETS, get_num_classes, NormalizeLayer
from time import time
from math import ceil
import pickle

import argparse
import datetime
import os
import torch
import numpy as np
import clip
from clip_modeling import *


class ImageNetCLIPLinear(nn.Module):
    def __init__(self, model):
        super().__init__()

        self.model = model

        num_classes = 1000
        hidden_dim = 768
        self.classifier = MLPHead(num_classes, hidden_dim)
        self.load_classifier()
        self._freeze_encoder()
    
    def _freeze_encoder(self,):
        requires_grad_(self.model, False)
    
    def load_classifier(self):
        ckpt_path = 'head_ckpt/imagenet_clip_vit_L14_nn2_clf.pth'
        print(f"Loading clf head from {ckpt_path}")
        self.classifier.load_state_dict(torch.load(ckpt_path, map_location=self.classifier.linear.weight.device))
    
    def forward(self, image_input):
        features = self.model.encode_image(image_input).type(image_input.type())
        logits = self.classifier(features)
        return logits.softmax(dim=-1)  # rulin: added softmax to be in the same scale of similarity scores


base_classifier, preprocess = clip.load('ViT-L/14')
preprocess.transforms = preprocess.transforms[:-1]
base_classifier = ImageNetCLIPLinear(base_classifier.cuda())
normalize_clip = preprocess.transforms[-1]
normalize_layer = NormalizeLayer(normalize_clip.mean, normalize_clip.std)
base_classifier = torch.nn.Sequential(normalize_layer, base_classifier)
#NOTE: add denoiser here
base_classifier = base_classifier.eval().cuda()

dataset = get_dataset('imagenet', 'test', preprocess=preprocess)
num_classes = 1000

def _count_arr(arr: np.ndarray, length: int) -> np.ndarray:
    counts = np.zeros(length, dtype=int)
    for idx in arr:
        counts[idx] += 1
    return counts

num, sigma = 1000, 1.0
counts_list, top1_ratio_list = [], []
for i in range(len(dataset)):
    (x, label) = dataset[i]
    x = x.cuda()
    print(x)
    print(torch.max(x), torch.min(x))
    stop
    
    with torch.no_grad():
        counts = np.zeros(num_classes, dtype=int)

        batch = x.repeat((num, 1, 1, 1))
        noise = torch.randn_like(batch, device='cuda') * sigma

        predictions = base_classifier(batch + noise).argmax(1)
        counts += _count_arr(predictions.cpu().numpy(), num_classes)
        
        counts_list.append(counts)
        top1_ratio = counts.max()/counts.sum()
        top1_ratio_list.append(top1_ratio)

        log = f"Img{i}, label{label}, top-1={np.argmax(counts)}, n_noise={counts.sum()}, Top-1 ratio={top1_ratio}"
        print(log)
        with open(f'test_prompt_{sigma}_linear.txt', 'a') as f:
            f.write(log+'\n')
        

print(f"Finished training, averaged top-1 ratio is {sum(top1_ratio_list)/len(top1_ratio_list)}")
with open(f'top1_ration_list_{sigma}_linear.pkl', 'wb') as f:
    pickle.dump(top1_ratio_list, f)

"""
export IMAGENET_DIR=/l/users/hongyiwa/datasets/ILSVRC2012/
CUDA_VISIBLE_DEVICES=3 python test_text_prompt_consistency.py &
"""