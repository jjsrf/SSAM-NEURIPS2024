
import copy
import numpy as np
from numpy import linalg as LA
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets


class Visualizer():

    def __init__(self, model,seed=None):
        self.model = model
        self.seed = seed
        self.direction = {}

    def generate_random_direction(self):

        if self.seed is not None:
            np.random.seed(self.seed)

        for name, W in self.model.named_parameters():
            if 'weight' in name and 'bn' not in name :
                #print(name, W.detach().cpu().numpy().shape)
                self.direction[name] = np.random.normal(0, 1, W.detach().clone().cpu().numpy().shape)
        #return self.direction

    def get_direction(self):
        return self.direction

    def save_random_direction(self, filename=None):
        self.generate_random_direction()
        assert filename is not None, "No numpy save filename provided!"
        if self.seed is not None:
            filename = filename + '_{}'.format(self.seed)
        else:
            print("No seed in the random direction is assigned! For reproductivity, seed is needed!")
            exit()

        np.save(filename, self.direction)

        print("npy file saved at {}".format(filename))

    def load_direction(self, filename=None):
        assert filename is not None, "No numpy load filename provided!"
        data = np.load(filename, allow_pickle=True)
        direction = data.item()
        for key in direction:
            self.direction[key] = direction[key]

    def layer_normalize(self):
        assert self.direction is not None, "random direction not loaded, cannot normalize."
        for name, W in self.model.named_parameters():
            if name in self.direction:
                #print(name, W.shape, self.direction[name].shape)
                #print('rand norm', LA.norm(self.direction[name]))
                self.direction[name] /= LA.norm(self.direction[name])
                #print('rand norm by itself', LA.norm(self.direction[name]))
                W_np = W.detach().clone().cpu().numpy()
                W_norm = LA.norm(W_np)
                #print(W_norm)
                self.direction[name] *= W_norm
                #print('layer normalized', LA.norm(self.direction[name]))

    def filter_normalize(self):
        assert self.direction is not None, "random direction not loaded, cannot normalize."
        for name, W in self.model.named_parameters():
            if name in self.direction:
                num_filters = len(self.direction[name])
                #print(name, W.shape, num_filters)
                W_np = W.detach().clone().cpu().numpy()
                for filter_idx in range(num_filters):
                    self.direction[name][filter_idx] /= LA.norm(self.direction[name][filter_idx])
                    W_norm = LA.norm(W_np[filter_idx])
                    self.direction[name][filter_idx] *= W_norm

                    #print(LA.norm(self.direction[name][filter_idx]))
                    #print(LA.norm(W.detach().clone().cpu().numpy()[filter_idx]))


    def pertube_model(self, delta):
        with torch.no_grad():
            for name, W in self.model.named_parameters():
                if name in self.direction:
                    W += delta * torch.from_numpy(self.direction[name]).cuda()
        return self.model
