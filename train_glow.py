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

from models.Glow import *
from deepul.hw2_helper import *
from arguments import parsing


# Added DDP implementations
class GlowSolver(object):
    def __init__(self, learning_rate=1e-4, n_epochs=128, local_rank=None, args=None):
        self.weight_y = 0.5

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
        self.graph = Glow().cuda()
        self.graph = DDP(module=self.graph, device_ids=[self.local_rank], find_unused_parameters=True)
        
        self.optim = torch.optim.Adam(self.graph.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=self.optim, lr_lambda=lambda epoch:0.99**epoch)

    def create_loaders(self):
        train, test = get_q3_data() # NHWC
        train = np.transpose(train, axes=[0, 3, 2, 1])  # NCHW 20000 x 3 x 32 x 32
        test = np.transpose(test, axes=[0, 3, 2, 1])  # NCHW 6838 x 3 x 32 x 32
        
        # Sampler for DDP
        train_sampler = DistributedSampler(dataset=train) #, shuffle=True)
        test_sampler = DistributedSampler(dataset=test) #, shuffle=False)

        train_loader = DataLoader(train, batch_size=self.batch_size, sampler=train_sampler, drop_last=True, pin_memory=False)
        test_loader = DataLoader(test, batch_size=self.batch_size, sampler=test_sampler, drop_last=True, pin_memory=False)
        return train_loader, test_loader

    def train(self):
        train_losses = []
        val_losses = []
        for epoch_i in range(self.n_epochs):
            epoch_i += 1

            self.graph.train()
            self.batch_loss_history = []

            for batch_i, image in enumerate(tqdm(
                    self.train_loader, desc='Batch', leave=False)):

                batch_i += 1
                # [batch_size, 3, 32, 32]
                image = Variable(image).float().contiguous().cuda()
                y_onehot = None

                # at first time, initialize ActNorm
                if epoch_i == 0:
                    self.graph(image[:self.batch_size // len(self.devices), ...],
                               y_onehot[:self.batch_size // len(self.devices), ...] if y_onehot is not None else None)

                z, nll, y_logits = self.graph(x=image, y_onehot=y_onehot)

                loss_generative = Glow.loss_generative(nll)
                loss_classes = 40

                batch_loss = loss_generative + loss_classes * self.weight_y

                # backward
                self.optim.zero_grad()
                batch_loss.backward()

                self.optim.step()
                self.scheduler.step()

                batch_loss = float(batch_loss.data)
                self.batch_loss_history.append(batch_loss)
            epoch_loss = np.mean(self.batch_loss_history)
            tqdm.write(f'Epoch {epoch_i} Loss: {epoch_loss:.2f}')

            train_losses.append(epoch_loss)
            val_losses.append(self.get_loss(self.val_loader))
            
            if epoch_i % self.save_epoch_step == 0:
                self.save_model_module(os.path.join(self.output_dir, "q4_b_ckpt_{}.pt".format(epoch_i)))
                np.save(os.path.join(self.output_dir, "glow_train_losses_{}.npy".format(epoch_i)), np.array(train_losses))
                np.save(os.path.join(self.output_dir, "glow_val_losses_{}.npy".format(epoch_i)), np.array(val_losses))

        self.save_model_module(os.path.join(self.output_dir, "q4_b_ckpt_{}_model_module.pt".format(self.n_epochs)))
        return train_losses, val_losses

    def get_loss(self, loader):
        errors = []
        self.graph.eval()

        for image in loader:
            with torch.no_grad():
                image = Variable(image).float().contiguous().cuda()
                y_onehot = None

                z, nll, y_logits = self.graph(x=image, y_onehot=y_onehot)

                loss_generative = Glow.loss_generative(nll)
                loss_classes = 0

                loss = loss_generative + loss_classes * self.weight_y
                error = float(loss.data)
                errors.append(error)
        log_string = f'Calc done! | '
        log_string += f'Loss: {np.mean(errors):.2f}'
        tqdm.write(log_string)
        return np.mean(errors)

    def sample(self, num_samples):
        with torch.no_grad():
            raw_samples = self.graph.module.sample(num_samples).cpu()
            samples = self.preprocess(raw_samples, reverse=True)
            return samples.cpu().numpy()

    def interpolate(self):
        self.graph.eval()
        good = [5, 13, 16, 19, 22]
        indices = []
        for index in good:
            indices.append(index*2)
            indices.append(index*2+1)
        with torch.no_grad():
            actual_images = next(iter(self.val_loader))[indices].to('cpu')
            assert actual_images.shape[0] % 2 == 0
            logit_actual_images, _ = self.preprocess(actual_images.float(), dequantize=False)
            latent_images, _ = self.graph.module.f(logit_actual_images)
            latents = []
            for i in range(0, actual_images.shape[0], 2):
                a = latent_images[i:i+1]
                b = latent_images[i + 1:i+2]
                diff = (b - a)/5.0
                latents.append(a)
                for j in range(1, 5):
                    latents.append(a + diff * float(j))
                latents.append(b)
            latents = torch.cat(latents, dim=0)
            logit_results = self.graph.module.g(latents)
            results = self.preprocess(logit_results, reverse=True)
            return results.cpu().numpy()

    def save_model_module(self, filename):
        torch.save(self.graph.module, filename)
        dist.barrier()

    def load_model(self, filename):
        torch.load(filename, map_location="cuda")


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
    
    solver = GlowSolver(n_epochs=args.num_epoch, args=args)
    solver.build()
    solver.train()

if __name__ == "__main__":
    main()

