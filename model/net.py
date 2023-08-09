"""Defines the neural network"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision.models.segmentation import deeplabv3_resnet50
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

def get_activation_func(activation):
    return  nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
        ['selu', nn.SELU(inplace=True)],
        ['sigmoid', nn.Sigmoid()],
        ['none', nn.Identity()]
    ])[activation]


def get_norm_layer(norm_layer, planes):
    return  nn.ModuleDict([
        ['bn', nn.BatchNorm2d(planes)],
        ['in', nn.InstanceNorm2d(planes)]
    ])[norm_layer]


def conv3x3(in_planes, out_planes, kernel_size=3, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

class BasicBlockTranser(nn.Module):

    def __init__(self, inplanes, middleplanes, outplanes, activation='relu', stride=1, dilation=1, norm_layer=None):
        super(BasicBlockTranser, self).__init__()
        self.conv1 = conv3x3(inplanes, middleplanes)
        self.bn1 = get_norm_layer(norm_layer, middleplanes)
        self.relu = get_activation_func(activation)
        self.conv2 = conv3x3(middleplanes, outplanes)
        self.bn2 = get_norm_layer(norm_layer, outplanes)

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        return out


class Transfer(nn.Module):
    def __init__(self, inplanes, middleplanes, outplanes, activation='relu', stride=1, dilation=1, norm_layer='bn', num_blocks=5):
        super(Transfer, self).__init__()
        layers = []
        layers.append(BasicBlockTranser(inplanes, middleplanes, middleplanes, activation, stride, dilation, norm_layer))
        for i in range(num_blocks-2):
            layers.append(BasicBlockTranser(middleplanes, middleplanes, middleplanes, activation, stride, dilation, norm_layer))
        layers.append(BasicBlockTranser(middleplanes, middleplanes, outplanes, activation, stride, dilation, norm_layer))
        self.transfer = nn.Sequential(*layers)

    def forward(self, x):
        x = self.transfer(x)
        return x

class AdaptiveNet(nn.Module):
    def __init__(self, encoder, transfer, decoder):
        super(AdaptiveNet, self).__init__()
        self.encoder = encoder
        self.transfer = transfer
        self.decoder = decoder

    def forward(self, x):
        x = self.encoder(x)
        x = self.transfer(x)
        x = self.decoder(x)
        return x


class AdaptiveNetCombined(nn.Module):
    def __init__(self, encoder1, transfer, encoder2, decoder):
        super(AdaptiveNetCombined, self).__init__()
        self.encoder1 = encoder1
        self.transfer = transfer
        self.encoder2 = encoder2
        self.decoder = decoder
        # self.combine = nn.Conv2d(38, 19, kernel_size=3, stride=1)
        
    def forward(self, x):
        
        z = self.encoder1(x)
        z = self.transfer(z)
        z = self.decoder(z)

        y = self.encoder2(x)
        y = self.decoder(y)
        # out = self.combine(torch.cat([y,z], dim=1))

        out = z*0.5 + y*0.5
        return out

# class AdaptiveNetCombined(nn.Module):
#     def __init__(self, source_encoder, source_decoder, cross_net):
#         super(AdaptiveNetCombined, self).__init__()
#         self.source_encoder = source_encoder
#         self.source_decoder = source_decoder
#         self.cross_net = cross_net
        
#     def forward(self, image):
        
#         x = self.source_encoder(image)
#         x = self.source_decoder(x)
#         x = torch.add(x, image)
#         out = self.cross_net(x)

#         return out

class Discriminator(nn.Module):

	def __init__(self, num_classes, ndf = 64):
		super(Discriminator, self).__init__()

		self.conv1 = nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1)
		self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1)
		self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)
		self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1)
		self.classifier = nn.Conv2d(ndf*8, 1, kernel_size=4, stride=2, padding=1)

		self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
		#self.up_sample = nn.Upsample(scale_factor=32, mode='bilinear')
		#self.sigmoid = nn.Sigmoid()


	def forward(self, x):
		x = self.conv1(x)
		x = self.leaky_relu(x)
		x = self.conv2(x)
		x = self.leaky_relu(x)
		x = self.conv3(x)
		x = self.leaky_relu(x)
		x = self.conv4(x)
		x = self.leaky_relu(x)
		x = self.classifier(x)
		#x = self.up_sample(x)
		#x = self.sigmoid(x) 

		return x

def load_weights(net, saved_state_dict):
    new_params = net.state_dict().copy()
    for name, param in new_params.items():
        if name in saved_state_dict and param.size() == saved_state_dict[name].size():
            new_params[name].copy_(saved_state_dict[name])
        else:
            print(name, "not copied")
    
    for name, param in net.state_dict().items():
        if name not in new_params:
            print(name, "not copied")
    
    net.load_state_dict(new_params)
    return net

def get_network(params):
    if params.architecture == 'deeplab_resnet101':
        net = deeplabv3_resnet101()
    if params.architecture == 'deeplab_resnet50':
        net = deeplabv3_resnet50()
        saved_state_dict = load_state_dict_from_url("https://download.pytorch.org/models/deeplabv3_resnet50_coco-cd0a2569.pth", progress=True)
        net = load_weights(net, saved_state_dict)

    if 'activation' in params.dict:
        net.classifier[-1] = nn.Sequential(nn.Conv2d(256, params.num_classes, 1, 1 ), get_activation_func(params.activation))
    else:
        net.classifier[-1] = nn.Conv2d(256, params.num_classes, 1, 1 )
    return net

def get_transfer(inplanes=2048, middleplanes=1024, outplanes=2048):
    return Transfer(inplanes, middleplanes, outplanes)

def get_adaptive_network(encoder, transfer, decoder):
    return AdaptiveNet(encoder, transfer, decoder)

def get_adaptive_network_combined(encoder1, transfer, encoder2, decoder):
    return AdaptiveNetCombined(encoder1, transfer, encoder2, decoder)

# def get_adaptive_network_combined(source_encoder, source_decoder, cross_net):
#     return AdaptiveNetCombined(source_encoder, source_decoder, cross_net)


def get_discriminator(num_classes):
    return Discriminator(num_classes)

