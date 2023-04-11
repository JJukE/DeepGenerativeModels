import os

import numpy as np
import pickle
from tqdm import tqdm
import time

import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F

from torch.autograd import Variable
from torchvision.utils import save_image, make_grid

import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader #, random_split
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from models.MultiscaleRealNVP import *
from deepul.hw2_helper import *
from arguments import parsing


# Added DDP implementations
class Solver(object):
    def __init__(self, learning_rate=5e-4, n_epochs=128, local_rank=None, args=None):
        self.log_interval = 100
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.output_dir = args.output_dir
        self.batch_size = args.batch_size
        self.save_epoch_step = args.save_epoch_step
        
        self.train_loader, self.val_loader = self.create_loaders()
        self.n_batches_in_epoch = len(self.train_loader)
        
        # for DDP
        self.local_rank = local_rank

    def build(self):
        self.flow = RealNVP().cuda()   
        self.flow = DDP(module=self.flow, device_ids=[self.local_rank]) # wrapping for DDP
        
        self.optimizer = torch.optim.Adam(self.flow.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=self.optimizer, lr_lambda=lambda epoch:0.99**epoch)

    def create_loaders(self):
        train, test = get_q3_data()
        train = np.transpose(train, axes=[0, 3, 1, 2])  # NCHW 20000 x 3 x 32 x 32
        test = np.transpose(test, axes=[0, 3, 1, 2])  # NCHW 6838 x 3 x 32 x 32
        
        # Sampler for DDP
        train_sampler = DistributedSampler(dataset=train) #, shuffle=True)
        test_sampler = DistributedSampler(dataset=test) #, shuffle=False)

        train_loader = DataLoader(train, batch_size=self.batch_size, sampler=train_sampler, pin_memory=False)
        test_loader = DataLoader(test, batch_size=self.batch_size, sampler=test_sampler, pin_memory=False)
        return train_loader, test_loader

    def preprocess(self, x, reverse=False, dequantize=True):
        if reverse:  # doesn't map back to [0, 4]
            x = 1.0 / (1 + torch.exp(-x))
            x -= 0.05
            x /= 0.9
            return x
        else:
            # dequantization
            if dequantize:
                x += torch.distributions.Uniform(0.0, 1.0).sample(x.shape).cuda()
            x /= 4.0

            # logit operation
            x *= 0.9
            x += 0.05
            logit = torch.log(x) - torch.log(1.0 - x)
            log_det = torch.nn.functional.softplus(logit) + torch.nn.functional.softplus(-logit) \
                      + torch.log(torch.tensor(0.9)) - torch.log(torch.tensor(4.0))
            return logit, torch.sum(log_det, dim=(1, 2, 3))

    def train(self):
        train_losses = []
        val_losses = []
        for epoch_i in range(self.n_epochs):
            epoch_i += 1

            self.flow.train()
            self.batch_loss_history = []

            for batch_i, image in enumerate(tqdm(
                    self.train_loader, desc='Batch', leave=False)):

                batch_i += 1
                # [batch_size, 3, 32, 32]
                image = Variable(image).cuda()
                logit_x, log_det = self.preprocess(image.float())
                log_prob = self.flow.module.log_prob(logit_x)
                log_prob += log_det

                batch_loss = -torch.mean(log_prob) / (3.0 * 32.0 * 32.0)

                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                batch_loss = float(batch_loss.data)
                self.batch_loss_history.append(batch_loss)

            epoch_loss = np.mean(self.batch_loss_history)
            tqdm.write(f'Epoch {epoch_i} Loss: {epoch_loss:.2f}')

            if epoch_i % self.save_epoch_step == 0:
                self.save_model_module(os.path.join(self.output_dir, "realnvp{}.model".format(str(epoch_i))))
            train_losses.append(epoch_loss)
            val_losses.append(self.get_loss(self.val_loader))
            np.save(os.path.join(self.output_dir, "train_losses.npy"), np.array(train_losses))
            np.save(os.path.join(self.output_dir, "val_losses.npy"), np.array(val_losses))

        self.save_model_module(os.path.join(self.output_dir, "q3_a_ckpt_{}_model_module.pt".format(self.n_epochs)))
        self.save_model_state_dict(os.path.join(self.output_dir, "q3_a_ckpt_{}_model_state_dict.pt".format(self.n_epochs)))
        self.save_model_module_state_dict(os.path.join(self.output_dir, "q3_a_ckpt_{}_model_module_state_dict.pt".format(self.n_epochs)))
        return train_losses, val_losses

    def save_model_module(self, filename):
        if self.local_rank == 0:
            torch.save(self.flow.module, filename)


class MultiSolver(Solver):
    def __init__(self, learning_rate=5e-4, n_epochs=128, local_rank=None, args=None):
        super().__init__(learning_rate, n_epochs, local_rank, args)

    def build(self):
        self.flow = MultiScaleRealNVP(in_channels=1, num_scales=4).cuda()
        self.flow = DDP(module=self.flow, device_ids=[self.local_rank])
        self.optimizer = torch.optim.Adam(self.flow.parameters(), lr=self.learning_rate)

    def train(self):
        train_losses = []
        val_losses = []
        for epoch_i in range(self.n_epochs):
            epoch_i += 1

            self.flow.train()
            self.batch_loss_history = []

            for batch_i, image in enumerate(tqdm(
                    self.train_loader, desc='Batch', leave=False)):

                batch_i += 1
                # [batch_size, 3, 32, 32]
                image = Variable(image).cuda()
                logit_x, log_det = self.preprocess(image.float())
                log_prob = self.flow.module.log_prob(logit_x)
                log_prob += log_det

                batch_loss = -torch.mean(log_prob) / (3.0 * 32.0 * 32.0)

                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()

                batch_loss = float(batch_loss.data)
                self.batch_loss_history.append(batch_loss)

            epoch_loss = np.mean(self.batch_loss_history)
            tqdm.write(f'Epoch {epoch_i} Loss: {epoch_loss:.2f}')

            if epoch_i % self.save_epoch_step == 0:
                self.save_model_module(os.path.join(self.output_dir, "q4_a_ckpt_{}_model_module.pt".format(str(epoch_i))))
            train_losses.append(epoch_loss)
            val_losses.append(self.get_loss(self.val_loader))
            np.save(os.path.join(self.output_dir, "multi_train_losses.npy"), np.array(train_losses))
            np.save(os.path.join(self.output_dir, "multi_val_losses.npy"), np.array(val_losses))

        np.save(os.path.join(self.output_dir, "multi_train_losses.npy"), np.array(train_losses))
        np.save(os.path.join(self.output_dir, "multi_val_losses.npy"), np.array(val_losses))
        return train_losses, val_losses

def get_q3_data():
    args = parsing()
    data_dir = args.data_dir
    train_data, test_data = load_pickled_data(join(data_dir, 'celeb.pkl'))
    return train_data, test_data

def main():
    args = parsing()
    
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    num_gpus = torch.distributed.get_world_size()
    
    solver = MultiSolver(n_epochs=args.num_epoch, local_rank=local_rank, args=args)
    solver.build()
    solver.train()

if __name__ == "__main__":
    main()
