import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F

from torch.autograd import Variable
from torchvision.utils import save_image, make_grid

from models.RealNVP import *


class AffineHalfHalfTransform(nn.Module):
    def __init__(self, type):
        super(AffineHalfHalfTransform, self).__init__()
        self.mask = self.build_mask(type=type)
        self.scale = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.scale_shift = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.resnet = SimpleResnet()

    def build_mask(self, type):
        # if type == "top", the top half is 1s
        # if type == "bottom", the bottom half is 1s
        # if type == "left", left half is 1s
        # if type == right", right half is 1s
        assert type in {"top", "bottom", "left", "right"}
        if type == "bottom":
            mask = np.concatenate([np.zeros([1, 1, 16, 32]), np.ones([1, 1, 16, 32])], axis=2)
        elif type == "top":
            mask = np.concatenate([np.ones([1, 1, 16, 32]), np.zeros([1, 1, 16, 32])], axis=2)
        elif type == "left":
            mask = np.concatenate([np.ones([1, 1, 32, 16]), np.zeros([1, 1, 32, 16])], axis=3)
        elif type == "right":
            mask = np.concatenate([np.zeros([1, 1, 32, 16]), np.ones([1, 1, 32, 16])], axis=3)
        else:
            raise NotImplementedError
        return torch.tensor(mask.astype('float32')).cuda()

    def forward(self, x, reverse=False):
        # returns transform(x), log_det
        batch_size, n_channels, _, _ = x.shape
        mask = self.mask.repeat(batch_size, 1, 1, 1)
        x_ = x * mask

        log_s, t = self.resnet(x_).split(n_channels, dim=1)
        log_s = self.scale * torch.tanh(log_s) + self.scale_shift
        t = t * (1.0 - mask)
        log_s = log_s * (1.0 - mask)

        if reverse:  # inverting the transformation
            x = (x - t) * torch.exp(-log_s)
        else:
            x = x * torch.exp(log_s) + t
        return x, log_s


class BadRealNVP(nn.Module):
    def __init__(self):
        super(BadRealNVP, self).__init__()

        self.prior = torch.distributions.Normal(torch.tensor(0.).cuda(), torch.tensor(1.).cuda())
        self.transforms = nn.ModuleList([
            AffineHalfHalfTransform("top"),
            ActNorm(3),
            AffineHalfHalfTransform("bottom"),
            ActNorm(3),
            AffineHalfHalfTransform("left"),
            ActNorm(3),
            AffineHalfHalfTransform("right"),
            ActNorm(3),
            AffineHalfHalfTransform("top"),
            ActNorm(3),
            AffineHalfHalfTransform("bottom"),
            ActNorm(3),
            AffineHalfHalfTransform("left"),
            ActNorm(3),
            AffineHalfHalfTransform("right")
        ])

    def g(self, z):
        # z -> x (inverse of f)
        x = z
        for op in reversed(self.transforms):
            x, _ = op.forward(x, reverse=True)
        return x

    def f(self, x):
        # maps x -> z, and returns the log determinant (not reduced)
        z, log_det = x, torch.zeros_like(x)
        for op in self.transforms:
            z, delta_log_det = op.forward(z)
            log_det += delta_log_det
        return z, log_det

    def log_prob(self, x):
        z, log_det = self.f(x)
        return torch.sum(log_det, [1, 2, 3]) + torch.sum(self.prior.log_prob(z), [1, 2, 3])

    def sample(self, num_samples):
        z = self.prior.sample([num_samples, 3, 32, 32])
        return self.g(z)