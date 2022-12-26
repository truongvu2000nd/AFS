import argparse
import os
import cv2
import sys
from .style_extraction import StyleExtractionNet
from .face_parsing import FaceParsing
from .stylegan2.stylegan2 import StyleGAN2Generator as Generator
from .e4e import pSp
import torch
import math
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms.functional as tF

""" 
Adapt from One Shot Face Swapping on Megapixels (https://arxiv.org/abs/2105.04932) official repository:
https://github.com/zyainfal/One-Shot-Face-Swapping-on-Megapixels/blob/main/inference/inference.py
"""


def encode_segmentation(segmentation, no_neck=True):
    # parse = segmentation[:,:,0]

    face_part_ids = [1, 2, 3, 4, 5, 6, 7, 10, 11, 12] if no_neck else [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 13, 14]
    mouth_id = 10
    hair_id = 13
    face_map = torch.zeros_like(segmentation)
    mouth_map = torch.zeros_like(segmentation)
    hair_map = torch.zeros_like(segmentation)

    for valid_id in face_part_ids:
        valid_index = torch.where(segmentation==valid_id)
        face_map[valid_index] = 1
    valid_index = torch.where(segmentation==mouth_id)
    mouth_map[valid_index] = 1
    valid_index = torch.where(segmentation==hair_id)
    hair_map[valid_index] = 1

    out = torch.cat([face_map, mouth_map, hair_map], axis=1)
    return out


class SoftErosion(nn.Module):
    def __init__(self, kernel_size=15, threshold=0.6, iterations=1):
        super(SoftErosion, self).__init__()
        r = kernel_size // 2
        self.padding = r
        self.iterations = iterations
        self.threshold = threshold

        # Create kernel
        y_indices, x_indices = torch.meshgrid(torch.arange(0., kernel_size), torch.arange(0., kernel_size))
        dist = torch.sqrt((x_indices - r) ** 2 + (y_indices - r) ** 2)
        kernel = dist.max() - dist
        kernel /= kernel.sum()
        kernel = kernel.view(1, 1, *kernel.shape)
        self.register_buffer('weight', kernel)

    def forward(self, x):
        x = x.float()
        for i in range(self.iterations - 1):
            x = torch.min(x, F.conv2d(x, weight=self.weight, groups=x.shape[1], padding=self.padding))
        x = F.conv2d(x, weight=self.weight, groups=x.shape[1], padding=self.padding)

        mask = x >= self.threshold
        x[mask] = 1.0
        x[~mask] /= x[~mask].max()

        return x, mask


class AFS(nn.Module):
    def __init__(self, ckpt_e4e, ckpt_stylegan, ckpt_style_extraction, ckpt_face_parsing=None, device="cuda"):
        super().__init__()
        # Inference Parameters
        self.size = 1024
        print("Load e4e model:", ckpt_e4e)
        ckpt = torch.load(ckpt_e4e, map_location="cpu")
        opts = ckpt['opts']

        opts['checkpoint_path'] = ckpt_e4e
        opts = argparse.Namespace(**opts)

        self.encoder = pSp(opts).eval()
        self.style_extraction = StyleExtractionNet(size=256, n_latent=18, num_layers=2, act="lrelu").eval()
        if ckpt_style_extraction is not None:
            print("Load style extraction model:", ckpt_style_extraction)
            ckpts = torch.load(ckpt_style_extraction, map_location="cpu")
            msg = self.style_extraction.load_state_dict(ckpts, strict=True)
            print(msg)
            del ckpts

        self.generator = Generator(truncation=0.65, use_w=True, resolution=1024, device=device).eval()
        if ckpt_stylegan is not None:
            print("Load generator:", ckpt_stylegan)
            self.generator.load_model(ckpt_stylegan)

        self.face_parsing = FaceParsing()
        if ckpt_face_parsing is not None:
            self.face_parsing = FaceParsing()
            print("Load face parsing:", ckpt_face_parsing)
            ckpts = torch.load(ckpt_face_parsing, map_location="cpu")
            msg = self.face_parsing.load_state_dict(ckpts)
            print(msg)
            del ckpts

        self.smooth_mask = SoftErosion(kernel_size=17, threshold=0.9, iterations=7)

        self.encoder.eval()
        self.style_extraction.eval()
        self.generator.eval()

    def forward(self, src, tgt):
        swapped_face = self.swap(src, tgt)
        resized_tgt = F.interpolate(tgt, size=(256, 256))
        mask_tgt = self.face_parsing(resized_tgt)
        encoded_mask = encode_segmentation(mask_tgt)
        encoded_mask = F.interpolate(encoded_mask.float(), tgt.shape[-2:])
        swapped_face = self.postprocess(swapped_face, tgt, encoded_mask)
        return swapped_face

    def swap(self, source, target):
        with torch.no_grad():
            bs = source.size(0)
            ts = torch.cat([target, source], dim=0)
            ts = F.interpolate(ts, (256, 256))
            codes = self.encoder(ts)

            ws = codes[bs:]
            wt = codes[:bs]
            w_swapped = ws - self.style_extraction(ws) + self.style_extraction(wt)
            fake_swap, _ = self.generator.w_plus_forward(w_swapped, resize=False)
            return fake_swap

    def postprocess(self, swapped_face, target, target_mask):
        face_mask_tensor = target_mask[:, 0, None] + target_mask[:, 1, None]

        soft_face_mask, _ = self.smooth_mask(face_mask_tensor)
        result =  swapped_face * soft_face_mask + target * (1 - soft_face_mask)
        return result
