#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
from torch import nn 
import torch.nn.functional as F
## local imports
from .network_utils import *



class NatureConvBody(nn.Module):
    def __init__(self, in_channels=4, noisy_linear=False):
        super(NatureConvBody, self).__init__()
        self.feature_dim = 512
        self.conv1 = layer_init(nn.Conv2d(in_channels, 32, kernel_size=8, stride=4))
        self.conv2 = layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2))
        self.conv3 = layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1))
        if noisy_linear:
            self.fc4 = NoisyLinear(7 * 7 * 64, self.feature_dim)
        else:
            self.fc4 = layer_init(nn.Linear(7 * 7 * 64, self.feature_dim))
        self.noisy_linear = noisy_linear

    def reset_noise(self):
        if self.noisy_linear:
            self.fc4.reset_noise()

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = F.relu(self.conv2(y))
        y = F.relu(self.conv3(y))
        y = y.view(y.size(0), -1)
        y = F.relu(self.fc4(y))
        return y



class DDPGConvBody(nn.Module):
    def __init__(self, in_channels=4):
        super(DDPGConvBody, self).__init__()
        self.feature_dim = 39 * 39 * 32
        self.conv1 = layer_init(nn.Conv2d(in_channels, 32, kernel_size=3, stride=2))
        self.conv2 = layer_init(nn.Conv2d(32, 32, kernel_size=3))

    def forward(self, x):
        y = F.elu(self.conv1(x))
        y = F.elu(self.conv2(y))
        y = y.view(y.size(0), -1)
        return y



class FCBody(nn.Module):
    '''Fully connected layers'''
    def __init__(self, state_dim, hidden_units=(64, 64), gate=nn.ReLU, noisy_linear=False,
                 init_method='orthogonal', batch_norm=None):
        super(FCBody, self).__init__()
        self.gate = gate
        self.noisy_linear = noisy_linear
        dims = (state_dim,) + hidden_units
        self.feature_dim = dims[-1]

        self.layers = nn.ModuleList()
        for i,(dim_in,dim_out) in enumerate(zip(dims[:-1], dims[1:])):
            if noisy_linear:
                self.layers.append(NoisyLinear(dim_in, dim_out))
            else:
                self.layers.append(layer_init(nn.Linear(dim_in, dim_out), method=init_method))
            self.layers.append(gate())  ## activation
            if i==0 and batch_norm is not None:  ## normalize the output of the 1st layer
                self.layers.append(batch_norm(dim_out))
        

    def reset_noise(self):
        if self.noisy_linear:
            for layer in self.layers:
                if isinstance(layer, NoisyLinear):
                    layer.reset_noise()


    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return(x)



class DummyBody(nn.Module):
    def __init__(self, state_dim):
        super(DummyBody, self).__init__()
        self.feature_dim = state_dim

    def forward(self, x):
        return x
