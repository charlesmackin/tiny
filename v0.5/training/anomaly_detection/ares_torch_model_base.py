"""
 @file   keras_model.py
 @brief  Script for keras model definition
 @author Toshiki Nakamura, Yuki Nikaido, and Yohei Kawaguchi (Hitachi Ltd.)
 Copyright (C) 2020 Hitachi, Ltd. All right reserved.
"""

########################################################################
# import python-library
########################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from torch.distributions import Normal, Uniform


def noise(weights, noise_type, noise_std, device):
    """
    Adds noise to the weights depending on the args for MNT/ANT
    :param weights: uncorrupted weights
    :return: noisy weights
    """
    weights_noisy = weights

    if noise_std > 0:
        if noise_type == 'log-normal':
            with torch.no_grad():
                stddev = noise_std * torch.ones_like(weights)
                mean = torch.zeros_like(weights)
                nu = Normal(mean, stddev)
                noise = torch.exp(nu.sample())
                noise = noise.to(device)
        elif noise_type == 'normal':
            with torch.no_grad():
                stddev = noise_std * torch.ones_like(weights)
                stddev = stddev.float()
                noise = Normal(1.0, stddev).sample()
                noise = torch.clamp(noise, min=0).to(device)
        elif noise_type == 'uniform':
            with torch.no_grad():
                stddev = noise_std * torch.ones_like(weights)
                if noise_std <= 1:
                    noise = Uniform(1 - stddev, 1 + stddev).sample().to(device)
                else:
                    noise = Uniform(0, 2 * stddev).sample().to(device)
        elif noise_type == 'custom-log-normal':
            with torch.no_grad():
                stddev = noise_std * torch.ones_like(weights)
                mean = torch.zeros_like(weights)
                nu = Normal(mean, stddev)
                add_noise = nu.sample().to(device)  # put some additive noise
                noise = torch.exp(nu.sample())
                noise = noise.to(device)
            weights_noisy = weights_noisy + 0.2 * add_noise  # this indent is important! will break inside torch.no_grad()
        elif noise_type == 'additive-noise':
            # 1) THIS VERSION WORKS BEST
            # std = nn.Parameter(noise_std * torch.ones_like(weights))  # need to define std and mean as nn.Parameter for grad
            # mean = nn.Parameter(torch.zeros_like(weights))
            # dist = Normal(mean, std)
            # add_noise = dist.sample().to(device)   # rsample allows for statistical grad calculation
            # noise = torch.ones_like(weights)
            # weights_noisy = weights_noisy + add_noise # this indent is important! will break inside torch.no_grad()

            # NO GRAD VERSION
            with torch.no_grad():
                std = noise_std * torch.ones_like(weights)  # need to define std and mean as nn.Parameter for grad
                mean = torch.zeros_like(weights)
                dist = Normal(mean, std)
                add_noise = dist.sample().to(device)   # rsample allows for statistical grad calculation
                noise = torch.ones_like(weights).to(device)
            weights_noisy = weights_noisy + add_noise # this indent is important! will break inside torch.no_grad()

            # WAS PREVIOUSLY WORKING BEST
            # std = nn.Parameter(torch.tensor(noise_std))  # need to define std and mean as nn.Parameter for grad
            # mean = nn.Parameter(torch.tensor(0.0))
            # dist = Normal(mean, std)
            # add_noise = dist.rsample(sample_shape=weights.shape).to(device)  # rsample allows for statistical grad calculation
            # # add_noise = dist.sample(weights.shape).to(device)   # sample no grad calc
            # noise = torch.ones_like(weights).to(device)
            # weights_noisy = weights_noisy + add_noise

            # 2) THIS VERSION DOESN'T TRAIN AS WELL
            # weights_noisy = weights + torch.empty_like(weights).normal_(mean=0., std=noise_std)
            # noise = torch.ones_like(weights)

            # 3) THIS VERSION ALSO DOESN'T TRAIN AS WELL
            # with torch.no_grad():
            #   stddev = noise_std * torch.ones_like(weights)
            #   mean = torch.zeros_like(weights)
            #   nu = Normal(mean, stddev)
            #   add_noise = nu.sample().to(device)
            #   noise = torch.ones_like(weights)
            # weights_noisy = weights_noisy + add_noise # this indent is important! will break inside torch.no_grad()

        else:
            sys.exit("Unknown noise_type {}".format(noise_type))

        return weights_noisy * noise


class NoisyLinear(nn.Module):

    def __init__(self, layer, layer_type='fc', mult_noise_type='log-normal', mult_noise_stddev=0.0, device='cpu'):
        super(NoisyLinear, self).__init__()
        self.layer = layer
        self.layer_type = layer_type
        self.mult_noise_stddev = mult_noise_stddev
        self.mult_noise_type = mult_noise_type
        self.device = device

    def forward(self, x):
        if self.mult_noise_stddev > 0:
            # if args.noise_std > 0 or args.add_noise_fraction > 0:
            weight = self.layer.weight
            bias = self.layer.bias

            # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            weight_noisy = noise(weight, self.mult_noise_type, self.mult_noise_stddev, self.device)

            if bias is not None:
                bias_noisy = noise(bias, self.mult_noise_type, self.mult_noise_stddev, self.device)
            else:
                bias_noisy = None

            return F.linear(x, weight_noisy, bias=bias_noisy)

        else:
            return self.layer(x)


class CustomReLU(nn.Module):
    def __init__(self, max_z=6.):
        # defaults to ReLU6
        super(CustomReLU, self).__init__()
        self.max_z = max_z

    def forward(self, x):
        return torch.clamp(x, min=0, max=self.max_z)


class GaussianNoise(nn.Module):
    """Gaussian noise regularizer.

    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values.
    """

    def __init__(self, sigma=0.04, device='cuda', is_relative_detach=False):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.device = device
        self.noise = torch.tensor(0, dtype=torch.float32).to(self.device)

    def forward(self, x):
        if self.sigma > 0:
            with torch.no_grad():
                scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
                # scale = self.sigma
                sampled_noise = self.noise.repeat(*x.size()).normal_() * scale
            x = x + sampled_noise
        return x


class ActivationNoise(nn.Module):
    """Gaussian noise regularizer.

    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values.
    """

    def __init__(self, sigma=0.04, device='cuda', is_relative_detach=False):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.device = device

    def forward(self, x):
        if self.sigma > 0:
            with torch.no_grad():
                scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
                # scale = self.sigma
                sampled_noise = torch.ones_like(x).normal_().to(self.device) * scale
            x = x + sampled_noise
        return x


class noisy_autoencoder(torch.nn.Module):
    def __init__(self, param):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(noisy_autoencoder, self).__init__()

        n_layers = param["n_layers"]
        inputDim = param["n_mels"] * param["frames"]
        h_size = param["h_size"]
        c_size = param["c_size"]
        use_bias = bool(param["bias"])
        max_z = param["max_z"]

        self.dropout = torch.nn.Dropout(p=param["dropout_ratio"])
        self.noise_type = param["noise_type"]
        self.noise_std = param["noise_std"] * param["max_weight"]   # so that noise scales as a percentage of weight range
        self.device = param["device"]
        self.out_noise = param["out_noise"]
        self.n_bits = 8.

        inp_sizes = [inputDim] + [h_size] * (n_layers - 1)
        inp_sizes = inp_sizes + [c_size] + list(reversed(inp_sizes))[0:-1]
        out_sizes = [h_size] * (n_layers - 1) + [c_size]
        out_sizes = out_sizes + list(reversed(out_sizes))[1:] + [inputDim]

        dropouts = [self.dropout] * n_layers * 2
        layers = [nn.Linear(inp_size, out_size, bias=use_bias) for inp_size, out_size in zip(inp_sizes, out_sizes)]
        noise_layers = [NoisyLinear(layer,
                                    layer_type='fc',
                                    mult_noise_type=self.noise_type,
                                    mult_noise_stddev=self.noise_std,
                                    device=self.device)
                        for layer in layers]
        z_noise = [GaussianNoise(sigma=self.out_noise, device=self.device)] * n_layers * 2
        # z_noise = [ActivationNoise(sigma=self.out_noise, device=self.device)] * n_layers * 2

        # act_fxn = [nn.ReLU6()] * n_layers * 2
        act_fxn = [CustomReLU(max_z=max_z)] * n_layers * 2

        # lists = [noise_layers, z_noise, act_fxn]
        # lists = [dropouts, noise_layers, act_fxn]

        lists = [dropouts, noise_layers, z_noise, act_fxn]
        network = [val for tup in zip(*lists) for val in tup]
        self.network = nn.Sequential(*network[0:-1])  # make sure to remove ReLU from end!

        print(self.network)

    def adc(self, z):
        # return self.z_max * torch.floor((z / self.z_max) * 2**self.n_bits) / 2**self.n_bits
        return z

    def custom_relu(self, x):
        return F.relu(x)
        # return torch.clip(F.relu(x), 0., self.z_max)

    def forward(self, x):
        return self.network(x)


def load_model(file_path):
    return torch.load(file_path)
