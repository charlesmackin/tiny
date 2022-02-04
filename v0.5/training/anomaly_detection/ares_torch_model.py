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
import torch.nn.functional as F
import sys
from torch.distributions import Normal, Uniform

def noise(weights, mult_noise_stddev, mult_noise_type, device):
  """
  Adds noise to the weights depending on the args for MNT/ANT
  :param weights: uncorrupted weights
  :return: noisy weights
  """
  weights_noisy = weights

  if mult_noise_stddev > 0:
    if mult_noise_type == 'log-normal':
      with torch.no_grad():
        stddev = mult_noise_stddev * torch.ones_like(weights)
        mean = torch.zeros_like(weights)
        nu = Normal(mean, stddev)
        noise = torch.exp(nu.sample())
        noise = noise.to(device)
    elif mult_noise_type == 'normal':
      with torch.no_grad():
        stddev = mult_noise_stddev * torch.ones_like(weights)
        stddev = stddev.float()
        noise = Normal(1.0, stddev).sample()
        noise = torch.clamp(noise, min=0).to(device)
    elif mult_noise_type == 'uniform':
     with torch.no_grad():
        stddev = mult_noise_stddev * torch.ones_like(weights)
        if mult_noise_stddev <=1:
          noise = Uniform(1-stddev, 1+stddev).sample().to(device)
        else:
          noise = Uniform(0, 2*stddev).sample().to(device)
    else:
      sys.exit("Unknown noise_type {}".format(mult_noise_type))

    weights_noisy = weights_noisy * noise

  return weights_noisy

def noise_layer(x, layer, layer_type='fc', mult_noise_stddev=0.0, mult_noise_type='log-normal', device='cpu'):
  """
  Adds noise to layer for MNT/ANT
  :param x: input activations
  :param layer: layer
  :param layer_type: type of layer: fc/conv2d
  :return: output activations of the layer with noisy weights
  """
  if mult_noise_stddev > 0:
  #if args.noise_std > 0 or args.add_noise_fraction > 0:
    weight = layer.weight
    bias = layer.bias

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    weight_noisy = noise(weight, mult_noise_stddev, mult_noise_type, device)

    if bias is not None:
      bias_noisy = noise(bias, mult_noise_stddev, mult_noise_type, device)
    else:
      bias_noisy = None

    if layer_type == 'fc':
      return F.linear(x, weight_noisy, bias=bias_noisy)
    elif layer_type == 'conv2d':
      return F.conv2d(x, weight_noisy, bias=bias_noisy, stride=layer.stride, padding=layer.padding)
    else:
      sys.exit("Unknown type {} in noise_layer".format(type))
  else:
    return layer(x)

class noisy_autoencoder(torch.nn.Module):
    def __init__(self, param):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """

        inputDim = param["feature"]["n_mels"] * param["feature"]["frames"]
        h_size = param["feature"]["h_size"]
        c_size = param["feature"]["c_size"]
        use_bias = True

        super(noisy_autoencoder, self).__init__()
        self.linear1 = torch.nn.Linear(inputDim, h_size, bias=use_bias)
        self.linear2 = torch.nn.Linear(h_size, h_size, bias=use_bias)
        self.linear3 = torch.nn.Linear(h_size, h_size, bias=use_bias)
        self.linear4 = torch.nn.Linear(h_size, h_size, bias=use_bias)
        self.linear5 = torch.nn.Linear(h_size, c_size, bias=use_bias)
        self.linear6 = torch.nn.Linear(c_size, h_size, bias=use_bias)
        self.linear7 = torch.nn.Linear(h_size, h_size, bias=use_bias)
        self.linear8 = torch.nn.Linear(h_size, h_size, bias=use_bias)
        self.linear9 = torch.nn.Linear(h_size, h_size, bias=use_bias)
        self.linear10 = torch.nn.Linear(h_size, inputDim, bias=use_bias)
        self.dropout = torch.nn.Dropout(p=param["feature"]["dropout"])

        self.mult_noise = 0.02
        self.mult_type = 'log-normal'
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.out_noise = 0.02
        self.z_max = 1000.
        self.n_bits = 8.

    def adc(self, z):
        #return self.z_max * torch.floor((z / self.z_max) * 2**self.n_bits) / 2**self.n_bits
        return z

    def custom_relu(self, x):
        return F.relu(x)
        #return torch.clip(F.relu(x), 0., self.z_max)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """

        activation = self.custom_relu
        adc = self.adc

        h = self.dropout(x)
        h = noise_layer(h, self.linear1, 'fc', mult_noise_stddev=self.mult_noise, mult_noise_type=self.mult_type, device=self.device)
        h = h + self.out_noise * torch.randn(h.shape).to(self.device)
        h = activation(h)
        #h = adc(h)

        # h = self.dropout(h)
        # h = noise_layer(h, self.linear2, 'fc', noise_std=self.mult_noise, noise_type=self.mult_type, device=self.device)
        # h = h + self.out_noise * torch.randn(h.shape)
        # h = activation(h)
        # #h = adc(h)

        h = self.dropout(h)
        h = noise_layer(h, self.linear3, 'fc', mult_noise_stddev=self.mult_noise, mult_noise_type=self.mult_type, device=self.device)
        h = h + self.out_noise * torch.randn(h.shape).to(self.device)
        h = activation(h)
        #h = adc(h)

        h = self.dropout(h)
        h = noise_layer(h, self.linear4, 'fc', mult_noise_stddev=self.mult_noise, mult_noise_type=self.mult_type, device=self.device)
        h = h + self.out_noise * torch.randn(h.shape).to(self.device)
        h = activation(h)
        #h = adc(h)

        h = self.dropout(h)
        h = noise_layer(h, self.linear5, 'fc', mult_noise_stddev=self.mult_noise, mult_noise_type=self.mult_type, device=self.device)
        h = h + self.out_noise * torch.randn(h.shape).to(self.device)
        h = activation(h)
        #h = adc(h)

        h = self.dropout(h)
        h = noise_layer(h, self.linear6, 'fc', mult_noise_stddev=self.mult_noise, mult_noise_type=self.mult_type, device=self.device)
        h = h + self.out_noise * torch.randn(h.shape).to(self.device)
        h = activation(h)
        #h = adc(h)

        h = self.dropout(h)
        h = noise_layer(h, self.linear7, 'fc', mult_noise_stddev=self.mult_noise, mult_noise_type=self.mult_type, device=self.device)
        h = h + self.out_noise * torch.randn(h.shape).to(self.device)
        h = activation(h)
        #h = adc(h)

        h = self.dropout(h)
        h = noise_layer(h, self.linear8, 'fc', mult_noise_stddev=self.mult_noise, mult_noise_type=self.mult_type, device=self.device)
        h = h + self.out_noise * torch.randn(h.shape).to(self.device)
        h = activation(h)
        #h = adc(h)

        # h = self.dropout(h)
        # h = noise_layer(h, self.linear9, 'fc', noise_std=self.mult_noise, noise_type=self.mult_type, device=self.device)
        # h = h + self.out_noise * torch.randn(h.shape)
        # h = activation(h)
        # #h = adc(h)

        h = self.dropout(h)
        h = noise_layer(h, self.linear10, 'fc', mult_noise_stddev=self.mult_noise, mult_noise_type=self.mult_type, device=self.device)
        h = h + self.out_noise * torch.randn(h.shape).to(self.device)
        #h = adc(h)

        return h

class autoencoder(torch.nn.Module):
    def __init__(self, inputDim):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        h_size = 128
        c_size = 8
        use_bias = True
        super(autoencoder, self).__init__()
        self.linear1 = torch.nn.Linear(inputDim, h_size, bias=use_bias)
        self.linear2 = torch.nn.Linear(h_size, h_size, bias=use_bias)
        self.linear3 = torch.nn.Linear(h_size, h_size, bias=use_bias)
        self.linear4 = torch.nn.Linear(h_size, h_size, bias=use_bias)
        self.linear5 = torch.nn.Linear(h_size, c_size, bias=use_bias)
        self.linear6 = torch.nn.Linear(c_size, h_size, bias=use_bias)
        self.linear7 = torch.nn.Linear(h_size, h_size, bias=use_bias)
        self.linear8 = torch.nn.Linear(h_size, h_size, bias=use_bias)
        self.linear9 = torch.nn.Linear(h_size, h_size, bias=use_bias)
        self.linear10 = torch.nn.Linear(h_size, inputDim, bias=use_bias)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        h = self.linear1(x)
        h = F.relu(h)

        h = self.linear2(h)
        h = F.relu(h)

        h = self.linear3(h)
        h = F.relu(h)

        h = self.linear4(h)
        h = F.relu(h)

        h = self.linear5(h)
        h = F.relu(h)

        h = self.linear6(h)
        h = F.relu(h)

        h = self.linear7(h)
        h = F.relu(h)

        h = self.linear8(h)
        h = F.relu(h)

        h = self.linear9(h)
        h = F.relu(h)

        h = self.linear10(h)

        return h

def load_model(file_path):
    return torch.load(file_path)


