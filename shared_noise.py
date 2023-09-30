# Code in this file is copied and adapted from
# https://github.com/ray-project/ray/tree/master/python/ray/rllib/es

import ray
import numpy as np
import torch

@ray.remote
def create_shared_noise():
    """
    Create a large array of noise to be shared by all workers. Used 
    for avoiding the communication of the random perturbations delta.
    """

    seed = 12345
    count = 250000000
    noise = np.random.RandomState(seed).randn(count).astype(np.float64)
    return noise


class SharedNoiseTable(object):
    def __init__(self, noise, seed = 11):

        self.rg = np.random.RandomState(seed)
        self.noise = noise
        assert self.noise.dtype == np.float64

    def get(self, i, dim):
        return self.noise[i:i + dim]

    def sample_index(self, dim):
        return self.rg.randint(0, len(self.noise) - dim + 1)

    def sample_indexes(self, dim, number):
        return self.rg.randint(0, len(self.noise) - dim + 1, number)

    def get_delta(self, dim):
        idx = self.sample_index(dim)
        return idx, self.get(idx, dim)

    def get_deltas(self,dim, number):
        idxes = self.sample_indexes(dim, number)
        noise = np.stack([self.get(i, dim) for i in idxes])
        return idxes, noise



class SharedNoiseTableSet(object):
    def __init__(self, noise, weights, seed = 11):

        self.rg = np.random.RandomState(seed)
        self.noise = torch.tensor(noise, dtype=torch.float64)
        self.W , self.bias = weights
        self.dim = 0
        for w in self.W:
            self.dim += w.numel()
        for b in self.bias:
            self.dim += b.numel()

        assert self.noise.dtype == torch.float64

    def get(self, i):
        idx = i
        W_noise = []
        bias_noise = []
        for w in self.W:
            dim = w.numel()
            W_noise.append(self.noise[idx:idx+dim].reshape(*w.shape))
            idx += dim
        for b in self.bias:
            dim = b.numel()
            bias_noise.append(self.noise[idx:idx+dim].reshape(*b.shape))
            idx += dim

        return [W_noise, bias_noise]

    def sample_index(self):
        return self.rg.randint(0, len(self.noise) - self.dim + 1)

    def sample_indexes(self, number):
        return self.rg.randint(0, len(self.noise) - self.dim + 1, number)

    def get_delta(self):
        idx = self.sample_index()
        return idx, self.get(idx)

    def get_deltas(self, number):
        idxes = self.sample_indexes(number)
        noise = [self.get(i) for i in idxes]
        return idxes, noise