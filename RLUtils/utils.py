import torch
import random
import collections
import torch.nn.functional as F


def freeze_network(network):
    for i in network:
        if torch.is_tensor(i):
            i.requires_grad = False


def unfreeze_network(network):
    for i in network:
        if torch.is_tensor(i):
            i.requires_grad = True


def forward(network, input_):
    for i in network:
        if torch.is_tensor(i):
            input_ = input_.matmul(i)
        else:
            input_ = i(input_)
    return input_


def create_network(dims):
    network = []
    for i, j in enumerate(dims):
        if i != len(dims) - 1:
            network.append(torch.randn([dims[i], dims[i + 1]], requires_grad=True))
            if i < len(dims) - 2:
                network.append(F.relu_)

    return network


class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = collections.deque(maxlen=buffer_size)

    def __call__(self, s, a, r, s_, d):
        # add (s,a,r,s',d) to the buffer
        self.buffer.append((s, a, r, s_, d))

    def sample(self, size):
        # return a sample of size B
        batch = random.sample(self.buffer, size)
        return list(map(torch.FloatTensor, zip(*batch)))
