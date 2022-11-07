"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
import tempfile
import math

import numpy as np
import torch as th
import torch.nn as nn
import torch.distributed as dist
import torchvision

from improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

def csc(x):
    return 1/math.sin(x)

def inv_csc(x):
    return math.asin(1/x)

def alpha_bar_t(t):
    def f_t(t):
        return th.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
    return f_t(t) / f_t(th.tensor(0))

def compute_time_step(sigma):
    s = 0.008
    T = 4000
    result = inv_csc(math.sqrt(1+sigma**2)*csc(math.pi/(2+2*s)))
    result = T*(1-(2*(1+s)*result)/math.pi)
    return int(result)

def plot_tensor(sample, path):
    sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
    transform = torchvision.transforms.ToPILImage()
    x_0 = transform(sample)
    x_0.save(path)


# class DiffusionDenoiser(nn.Module):
#     def __init__(self, args, preprocess):
#         super(DiffusionDenoiser, self).__init__()
#         self.model, diffusion = create_model_and_diffusion(
#             **args_to_dict(args, model_and_diffusion_defaults().keys())
#         )
#         self.model.load_state_dict(
#             th.load(args.model_path, map_location="cpu")
#         )
#         self.model.eval()

#         self.sample_fn = diffusion.p_mean_variance
#         self.t_star = th.tensor(compute_time_step(args.sigma)).cuda()
#         self.alpha_bar_star = alpha_bar_t(self.t_star)

#         self.to_pil = torchvision.transforms.ToPILImage()
#         self.preprocess = preprocess
    
#     def forward(self, x_batch):
#         x = x_batch * 2 - 1 # assume inputs of range [0,1]
#         bs = x.shape[0]
#         sample = self.sample_fn(
#             self.model,
#             x,
#             self.t_star.repeat((bs)),
#             clip_denoised=True,
#             model_kwargs={},
#         )["pred_xstart"]
        
#         batch = []
#         for i in range(bs):
#             x = sample[i]
#             x = ((x + 1) * 127.5).clamp(0, 255).to(th.uint8)
#             x = self.to_pil(x)
#             x = self.preprocess(x)
#             batch.append(x)
#         return  th.stack(batch).cuda()


class DiffusionDenoiser4ViT(nn.Module):
    def __init__(self, args):
        super(DiffusionDenoiser4ViT, self).__init__()
        self.model, diffusion = create_model_and_diffusion(
            **args_to_dict(args, model_and_diffusion_defaults().keys())
        )
        self.model.load_state_dict(
            th.load(args.model_path, map_location="cpu")
        )
        self.model.eval()

        self.sample_fn = diffusion.p_mean_variance
        self.t_star = th.tensor(compute_time_step(args.sigma)).cuda()
        alpha_bar_star = alpha_bar_t(self.t_star)
        self.sqrt_alpha_bar_star = th.sqrt(alpha_bar_star)
        print(args.sigma, self.t_star, alpha_bar_star)

        self.resize = torchvision.transforms.Resize(224)
    
    def forward(self, x_batch):
        x = x_batch * self.sqrt_alpha_bar_star  # x_batch: (x*2-1) + N(0, sigma)
        bs = x.shape[0]
        sample = self.sample_fn(
            self.model,
            x,
            self.t_star.repeat((bs)),
            clip_denoised=True,
            model_kwargs={},
        )["pred_xstart"]  # estimation of x0 \in [-1,1]
        sample = (sample + 1) / 2  # rescale to [0, 1]
        return self.resize(sample)


def main():
    th.distributed.init_process_group("nccl", world_size=1, rank=0)#, export MASTER_ADDR=192.168.62.121, export MASTER_PORT=12000

    args = create_argparser().parse_args()

    print("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        th.load(args.model_path, map_location="cpu")
    )
    # model.cuda()
    model.eval()

    print("denoising...")
    dataset = torchvision.datasets.CIFAR10(train=False, root='~/data/cifar10', download=True)
    x = dataset[0][0].convert("RGB")
    x.save('output/original.png')
    x = np.array(x)
    x = x.astype(np.float32) / 127.5 - 1
    x = np.transpose(x, [2, 0, 1])
    x = np.expand_dims(x, axis=0)
    x = th.from_numpy(x)

    sigma = 1.0
    noise = th.randn_like(x) * sigma
    x = x + noise
    plot_tensor(x[0], 'output/noisy.png')
    t = np.array(compute_time_step(sigma))
    t = np.expand_dims(t, axis=0)
    t = th.from_numpy(t)
    print(t)
    
    sample_fn = diffusion.p_mean_variance
    sample = sample_fn(
        model,
        x,
        t,
        clip_denoised=args.clip_denoised,
        model_kwargs={},
    )["pred_xstart"]
    plot_tensor(sample[0], 'output/denoised.png')


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=100,
        batch_size=16,
        use_ddim=False,
        model_path="/home/dacheng.li/checkpoints/diffusion/cifar10_uncond_50M_500K.pt",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
