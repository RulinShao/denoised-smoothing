import os
from turtle import forward
import clip
import torch
import torch.nn as nn

import numpy as np
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle


def requires_grad_(model:torch.nn.Module, requires_grad:bool) -> None:
    for param in model.parameters():
        param.requires_grad_(requires_grad)


class LinearHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear1 = nn.Linear(512, 2048)
        self.act1 = nn.ReLU()
        self.linear2 = nn.Linear(2048, 1000)
    
    def forward(self, features):
        out = self.linear1(features)
        out = self.act1(out)
        out = self.linear2(out)
        return out

class LogisticRegression(nn.Module):
     def __init__(self):
         super(LogisticRegression, self).__init__()
         self.linear = torch.nn.Linear(512, 1000)
     def forward(self, x):
         outputs = torch.sigmoid(self.linear(x))
         return outputs


class CLIPVisionLinearProbing(nn.Module):
    def __init__(self, model, args=None):
        super().__init__()

        self.args = args
        self.model = model
        self.classifier = LinearHead()
        self.load_classifier()
        # self.classifier = nn.Linear(512, 1000)

        self._freeze_encoder()
    
    def _freeze_encoder(self,):
        requires_grad_(self.model, False)
    
    def load_classifier(self):
        assert os.path.isfile(self.args.clf_head_ckpt)
        print(f"Loading clf head from {self.args.clf_head_ckpt}")
        self.classifier.load_state_dict(torch.load(self.args.clf_head_ckpt, map_location=self.classifier.linear1.weight.device))
    
    def forward(self, image_input):
        features = self.model.encode_image(image_input).type(image_input.type())
        logits = self.classifier(features)
        return logits


class CLIPModelForZeroShotClassication(nn.Module):
    """
    given a set of prompting templates, 
    (orgianl) do average on all prompt embeddings
    (ours) we want to do predictions but do argmax over all templates TODO
    """
    def __init__(self, model, classes, preprocess=None, args=None):
        super().__init__()
        self.model = model
        if args.dataset == 'imagenet':
            # from imagenet_utils import imagenet_templates
            # templates = imagenet_templates
            # NOTE: testing single prompting
            templates = ['a photo of {}.']
        else:
            templates = ['a photo of {}.']
        
        with torch.no_grad():
            text_features = []
            for classname in tqdm(classes):
                texts = [template.format(classname) for template in templates]
                texts = clip.tokenize(texts).cuda()
                class_embeddings = model.encode_text(texts)
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                text_features.append(class_embedding)
            self.text_features = torch.stack(text_features, dim=1).cuda()
            print(f"Text feature shape: {self.text_features.shape}")

    def forward(self, image_input):
        
        # Calculate features
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            
            # Compute similarity scores
            image_features /= image_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ self.text_features).softmax(dim=-1)
            return similarity


class CLIPDualStreamForClassification(nn.Module):
    def __init__(self, model, classes, preprocess=None, args=None):
        super().__init__()
        self.alpha = args.dual_alpha
        assert self.alpha >= 0 and self.alpha <=1
        if not self.alpha == 0.0:
            print(f"Initializing zero-shot classifier...")
            self.zero_shot_model = CLIPModelForZeroShotClassication(model, classes, preprocess, args)
        if not self.alpha == 1.0:
            print(f"Initializing linear probing classifier...")
            self.vision_clf_model = CLIPVisionLinearProbing(model, classes, preprocess, args)

    def forward(self, image_input):
        if not self.alpha == 0.0:
            zero_shot_predictions = self.zero_shot_model(image_input)
        else:
            zero_shot_predictions = 0.0
        
        if not self.alpha == 1.0:
            vision_clf_prefictions = self.vision_clf_model(image_input)
        else:
            vision_clf_prefictions = 0.0
        
        predictions = self.alpha * zero_shot_predictions + (1 - self.alpha) * vision_clf_prefictions
        return predictions