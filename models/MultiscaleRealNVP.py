import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F

from torch.autograd import Variable
from torchvision.utils import save_image, make_grid

from models.RealNVP import *


class RealNVPBlock(RealNVP):
    def __init__(self, in_channels, mid_channels):
        super().__init__()
        
        self.s = nn.Sequential(
            nn.Conv2d(in_channels//2, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(mid_channels, in_channels//2, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

        self.t = nn.Sequential(
            nn.Conv2d(in_channels//2, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(mid_channels, in_channels//2, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x, reverse=False):
        x1, x2 = x.chunk(2, dim=1)
        s = self.s(x1)
        t = self.t(x1)
        if not reverse:
            y2 = (x2 + t) * torch.exp(-s)
        else:
            y2 = x2 * torch.exp(s) - t
        y = torch.cat((x1, y2), dim=1)
        log_det = torch.sum(s, dim=(1, 2, 3))
        return y, log_det

class ChannelMask(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.num_channels = num_channels
        # self.register_buffer("mask", self.create_mask())

    def create_mask(self):
        mask = torch.zeros(self.num_channels)
        mask[: self.num_channels // 2] = 1
        return mask.view(1, self.num_channels, 1, 1)

    def forward(self, x):
        z = x * self.mask.to(x.device)
        log_det = torch.zeros(x.shape[0], device=x.device)
        return z, log_det

    def inverse(self, z):
        x = z / self.mask.to(z.device)
        log_det = torch.zeros(z.shape[0], device=z.device)
        return x, log_det

    def mask(self, x):
        return self.forward(x)

    def unmask(self, z):
        return self.inverse(z)


class CheckerboardMask(nn.Module):
    def __init__(self):
        super().__init__()

    def mask(self, x):
        b, c, h, w = x.shape
        mask = torch.zeros(b, 1, h, w, device=x.device)
        mask[:, :, ::2, ::2] = 1
        mask[:, :, 1::2, 1::2] = 1
        mask = mask.expand(b, c // 4, h, w)
        z_masked = x * mask
        z_unmasked = x * (1 - mask)
        return z_unmasked, z_masked

    def unmask(self, z, z_masked):
        return z * (1 - z_masked) + z_masked


class MultiScaleRealNVP(nn.Module):
    def __init__(self, in_channels, num_scales, mid_channels=64):
        super().__init__()
        
        self.in_channels = in_channels
        self.num_scales = num_scales
        
        self.checkerboard_masks = nn.ModuleList()
        self.channel_masks = nn.ModuleList()
        self.transforms = nn.ModuleList()
        
        for i in range(num_scales):
            checkerboard_mask = CheckerboardMask().cuda()
            channel_mask = ChannelMask(in_channels // 2).cuda()
            transform = RealNVPBlock(in_channels // 2, mid_channels).cuda()
            
            self.checkerboard_masks.append(checkerboard_mask)
            self.channel_masks.append(channel_mask)
            self.transforms.append(transform)
            
            in_channels //= 2
        
        self.mean = nn.Parameter(torch.zeros(3, 64, 64))
        self.log_std = nn.Parameter(torch.zeros(3, 64, 64))

    def forward(self, x):
        log_det = torch.zeros(x.shape[0], device=device)
        z = x
        
        for i in range(self.num_scales):
            z, log_det = self.scale_forward(z, log_det, i)
            
        return z, log_det

    def inverse(self, z):
        x = z
        
        for i in reversed(range(self.num_scales)):
            x = self.scale_inverse(x, i)
            
        return x
    
    def scale_forward(self, z, log_det, scale_idx):
        checkerboard_mask = self.checkerboard_masks[scale_idx]
        channel_mask = self.channel_masks[scale_idx]
        transform = self.transforms[scale_idx]
        
        z, z_masked = checkerboard_mask.mask(z)
        z, log_det_1 = channel_mask.mask(z)
        z, log_det_2 = transform.forward(z)
        log_det += log_det_1 + log_det_2
        
        if scale_idx < self.num_scales - 1:
            z_next, z_top = torch.chunk(z, 2, dim=1)
            log_det, z_next = self.scale_forward(z_next, log_det, scale_idx + 1)
            z = torch.cat([z_next, z_top], dim=1)
        
        z = checkerboard_mask.unmask(z, z_masked)
        return z, log_det
    
    def scale_inverse(self, x, scale_idx):
        checkerboard_mask = self.checkerboard_masks[scale_idx]
        channel_mask = self.channel_masks[scale_idx]
        transform = self.transforms[scale_idx]
        
        x, x_masked = checkerboard_mask.mask(x)
        
        if scale_idx < self.num_scales - 1:
            x_next, x_top = torch.chunk(x, 2, dim=1)
            x_next = self.scale_inverse(x_next, scale_idx + 1)
            x = torch.cat([x_next, x_top], dim=1)
        
        x, _ = transform.inverse(x)
        x, _ = channel_mask.unmask(x)
        x = checkerboard_mask.unmask(x, x_masked)
        
        return x