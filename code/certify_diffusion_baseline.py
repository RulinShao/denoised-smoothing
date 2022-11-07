# evaluate a smoothed classifier on a dataset
from architectures import get_architecture, IMAGENET_CLASSIFIERS
from core import Smooth, SmoothOptimizeAlpha
from datasets import get_dataset, DATASETS, get_num_classes, NormalizeLayer
from time import time
from tqdm import tqdm

import argparse
import datetime
import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForImageClassification

from stable_diffusion.image_denoise import *


defaults = dict(
    clip_denoised=True,
    num_samples=100,
    batch_size=16,
    use_ddim=False,
)
defaults.update(model_and_diffusion_defaults())
parser = argparse.ArgumentParser(description='Certify many examples')
add_dict_to_argparser(parser, defaults)

parser.add_argument("--dataset", default='imagenet', choices=DATASETS, help="which dataset")
parser.add_argument("--clf_head_ckpt", type=str, default="head_ckpt/imagenet_clip_vit_L14_nn2_clf.pth", 
                    help="path to save or saved sklearn classifier")
parser.add_argument("--model_type", type=str, default='ViT-L/14')
parser.add_argument("--dual_alpha", type=float, default=1.0)
parser.add_argument("--optimize_alpha", action="store_true")
parser.add_argument("--clip_alpha_split_num", type=int, default=10)
parser.add_argument("--sigma", type=float, help="noise hyperparameter")
parser.add_argument("--outfile", type=str, help="output file")
parser.add_argument("--batch", type=int, default=1000, help="batch size")
parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument("--N0", type=int, default=100)
parser.add_argument("--N", type=int, default=10000, help="number of samples to use")
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
parser.add_argument('--denoiser', type=str, default='',
                    help='Path to a denoiser to attached before classifier during certificaiton.')
parser.add_argument('--azure_datastore_path', type=str, default='',
                    help='Path to imagenet on azure')
parser.add_argument('--philly_imagenet_path', type=str, default='',
                    help='Path to imagenet on philly')
parser.add_argument('--evaluate_sanity', action='store_true')

parser.add_argument('--model_path', default='/home/dacheng.li/checkpoints/diffusion/cifar10_uncond_50M_500K.pt')
args = parser.parse_args()

args.image_size = 32
args.num_channels = 128
args.num_res_blocks = 3
args.learn_sigma = True
args.dropout = 0.3
args.diffusion_steps = 4000
args.noise_schedule = 'cosine'
args.model_path = '/home/dacheng.li/checkpoints/diffusion/cifar10_uncond_50M_500K.pt'

if args.azure_datastore_path:
    os.environ['IMAGENET_DIR_AZURE'] = os.path.join(args.azure_datastore_path, 'datasets/imagenet_zipped')
elif args.philly_imagenet_path:
    os.environ['IMAGENET_DIR_PHILLY'] = os.path.join(args.philly_imagenet_path, './')
else:
    os.environ['IMAGENET_DIR_PHILLY'] = "/hdfs/public/imagenet/2012/"


class HF_Wrapper(nn.Module):
    def __init__(self, model):
        super(HF_Wrapper, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)['logits']


if __name__ == "__main__":
    classes = get_dataset(args.dataset, 'test').classes if args.dataset == 'cifar10' else None
    
    # load the base classifier
    base_classifier = AutoModelForImageClassification.from_pretrained("aaraki/vit-base-patch16-224-in21k-finetuned-cifar10")
    base_classifier = HF_Wrapper(base_classifier)
    
    # add denoiser
    denoiser = DiffusionDenoiser4ViT(args=args)
    base_classifier = torch.nn.Sequential(denoiser, base_classifier)

    # set to eval mode
    base_classifier = base_classifier.eval().cuda()

    # iterate through the dataset
    dataset = get_dataset(args.dataset, 'test')
    
    if args.evaluate_sanity:
    # test sanity accuracy
        num_correct = num_samples = 0
        with torch.no_grad():
            for images, labels in tqdm(DataLoader(dataset, batch_size=100)):
                images, labels = images.cuda(), labels.cuda()
                predictions = base_classifier(images)
                _, predictions = predictions.max(1)
                num_correct += (predictions == labels).sum()
                num_samples += predictions.size(0)
        print(f"Sanity Accuracy {float(num_correct)/float(num_samples)*100:.2f}")
    
    else:
        # create the smooothed classifier g
        if args.optimize_alpha:
            print("Using sample-wise alpha")
            smoothed_classifier = SmoothOptimizeAlpha(base_classifier, get_num_classes(args.dataset), args.sigma, args.clip_alpha_split_num, diffusion=True)
        else:
            smoothed_classifier = Smooth(base_classifier, get_num_classes(args.dataset), args.sigma, diffusion=True)

        # prepare output file
        if not os.path.exists(args.outfile.split('sigma')[0]):
            os.makedirs(args.outfile.split('sigma')[0])

        f = open(args.outfile, 'w')
        print("idx\tlabel\tpredict\tradius\tcorrect\ttime", file=f, flush=True)
        print("idx\tlabel\tpredict\tradius\tcorrect\ttime", flush=True)
        f.close()
        
        for i in range(len(dataset)):

            # only certify every args.skip examples, and stop after args.max examples
            if i % args.skip != 0:
                continue
            if i == args.max:
                break

            (x, label) = dataset[i]

            before_time = time()
            # certify the prediction of g around x
            x = x.cuda()
            prediction, radius = smoothed_classifier.certify(x, args.N0, args.N, args.alpha, args.batch)
            after_time = time()
            correct = int(prediction == label)

            time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
            
            f = open(args.outfile, 'a')
            print("{}\t{}\t{}\t{:.3}\t{}\t{}".format(
                i, label, prediction, radius, correct, time_elapsed), file=f, flush=True)
            print("{}\t{}\t{}\t{:.3}\t{}\t{}".format(
                i, label, prediction, radius, correct, time_elapsed), flush=True)
            f.close()
