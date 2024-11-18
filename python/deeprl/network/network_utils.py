#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from ..utils import *



class BaseNet:
    def __init__(self):
        pass

    def reset_noise(self):
        pass



# Adapted from https://github.com/saj1919/RL-Adventure/blob/master/5.noisy%20dqn.ipynb
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.4):
        super(NoisyLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.zeros((out_features, in_features)), requires_grad=True)
        self.weight_sigma = nn.Parameter(torch.zeros((out_features, in_features)), requires_grad=True)
        self.register_buffer('weight_epsilon', torch.zeros((out_features, in_features)))

        self.bias_mu = nn.Parameter(torch.zeros(out_features), requires_grad=True)
        self.bias_sigma = nn.Parameter(torch.zeros(out_features), requires_grad=True)
        self.register_buffer('bias_epsilon', torch.zeros(out_features))

        self.register_buffer('noise_in', torch.zeros(in_features))
        self.register_buffer('noise_out_weight', torch.zeros(out_features))
        self.register_buffer('noise_out_bias', torch.zeros(out_features))

        self.reset_parameters()
        self.reset_noise()

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma.mul(self.weight_epsilon)
            bias = self.bias_mu + self.bias_sigma.mul(self.bias_epsilon)
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.weight_mu.size(1))

        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.weight_sigma.size(1)))

        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.bias_sigma.size(0)))

    def reset_noise(self):
        self.noise_in.normal_(std=Config.NOISY_LAYER_STD)
        self.noise_out_weight.normal_(std=Config.NOISY_LAYER_STD)
        self.noise_out_bias.normal_(std=Config.NOISY_LAYER_STD)

        self.weight_epsilon.copy_(self.transform_noise(self.noise_out_weight).ger(
            self.transform_noise(self.noise_in)))
        self.bias_epsilon.copy_(self.transform_noise(self.noise_out_bias))

    def transform_noise(self, x):
        return x.sign().mul(x.abs().sqrt())




## add uniform methods etc., by nov05
def layer_init(layer, w_scale=1.0, 
               method='orthogonal', fr=0, to=1):
    if method=='orthogonal':
        nn.init.orthogonal_(layer.weight.data)
        layer.weight.data.mul_(w_scale)
        nn.init.constant_(layer.bias.data, 0)
    elif method=='uniform':
        layer.weight.data.uniform_(fr, to)  ## default (0,1)
    elif method=='uniform_fan_in':
        fan_in = layer.weight.data.size()[0]
        to = 1./np.sqrt(fan_in)
        fr = -to 
        layer.weight.data.uniform_(fr, to)
    else:
        raise NotImplementedError
    return layer



## added by nov05
def soft_update_network(target, source, tau):
    ## trg = trg*(1-τ) + src*τ
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data*(1.-tau) + source_param.data*tau)



## added by nov05
def check_network_params(network_name, network, raise_error=True):
    has_error = False
    for name, param in network.named_parameters():
        if torch.isnan(param).any():
            has_error = True
            print(f"⚠️ NaN detected in {network_name}.{name}")
        if torch.isinf(param).any():
            has_error = True
            print(f"⚠️ Inf detected in {network_name}.{name}")
    if raise_error and has_error:
        raise



## added by nov05
def check_tensor(tensor_name, tensor, raise_error=True):
    has_error = False
    if torch.isnan(tensor).any(): 
        has_error = True
        print(f"⚠️ {tensor_name} is NaN.")
    if torch.isinf(tensor).any():
        has_error = True
        print(f"⚠️ {tensor_name} is Inf.")
    if raise_error and has_error:
        raise