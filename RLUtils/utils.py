import torch
import random
import collections
import torch.nn.functional as F
import math
import torch.nn as nn


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


def create_network(dims,activation=F.relu):
    network = []
    for i, j in enumerate(dims):
        if i != len(dims) - 1:
            network.append(torch.randn([dims[i], dims[i + 1]], requires_grad=True))
            if i < len(dims) - 2:
                network.append(activation)

    return network


def create_network_with_nn(dims,activation=nn.ReLU()): # Have to put nn.ReLU() not nn.ReLU
    network = []
    for i, j in enumerate(dims):
        if i != len(dims) - 1:
            network.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                network.append(activation)

    return nn.Sequential(*network)


def create_network_end_activation(dims,activation=nn.ReLU(),output_activation=nn.Tanh()):
    network = []
    for i, j in enumerate(dims):
        if i != len(dims) - 1:
            network.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                network.append(activation)
    network.append(output_activation)

    return nn.Sequential(*network)


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


def appendOnes(tensor):
    '''
    Example
    a = torch.randn([3,3])
    b = torch.ones([3,1])
    torch.cat([b,a],1)
    Out[1]:
    tensor([[ 1.0000,  1.3271,  0.9431,  0.2787],
        [ 1.0000, -0.1420,  0.6311, -1.3694],
        [ 1.0000, -1.3565, -0.5244, -0.5571]])
    :param tensor: input tensor to append 1's
    :return: tensor appended with 1's
    '''
    c = tensor.shape
    assert len(c) == 2
    b = torch.ones([c[0], 1])
    tensor = torch.cat([b, tensor], 1)
    return tensor


def SGD(parameters, gradients, learning_rate):
    # TODO : implement momentum
    with torch.no_grad():
        parameters.data -= learning_rate * gradients
    return parameters


def MeanSquarredError(**kwargs):
    labels, targets, batch_size = kwargs['labels'], kwargs['targets'], kwargs['batch_size']
    return torch.sum(torch.mul((labels - targets), (labels - targets))) / batch_size


def kaiming_initialization(x, mode='out'):
    '''

    :param x: tensor
    :param mode: in or out
    Mode 'out' preserves variance of outputs in the forward pass
    Mode 'in' preserves variance of gradients in backward pass
    :return: tensor updated with initialization scheme
    '''

    # TODO : update for high rank tensors
    if len(x.size()) == 2:
        a, b = x.size()
    if len(x.size()) == 1:
        a = b = x.size()[0]
    if mode == 'out':
        x.data = x.data * math.sqrt(2 / a)
    else:
        x.data = x.data * math.sqrt(2 / b)
