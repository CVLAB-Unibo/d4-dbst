from torch import nn


def conv3x3(in_planes, out_planes, kernel_size=3, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=kernel_size,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


class BasicBlockTranser(nn.Module):
    def __init__(self, inplanes, middleplanes, outplanes):
        super(BasicBlockTranser, self).__init__()
        self.conv1 = conv3x3(inplanes, middleplanes)
        self.bn1 = nn.BatchNorm2d(middleplanes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(middleplanes, outplanes)
        self.bn2 = nn.BatchNorm2d(outplanes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        return out


class Transfer(nn.Module):
    def __init__(self, inplanes, middleplanes, outplanes, num_blocks=5):
        super(Transfer, self).__init__()
        layers = []
        layers.append(BasicBlockTranser(inplanes, middleplanes, middleplanes))
        for i in range(num_blocks - 2):
            layers.append(BasicBlockTranser(middleplanes, middleplanes, middleplanes))
        layers.append(BasicBlockTranser(middleplanes, middleplanes, outplanes))
        self.transfer = nn.Sequential(*layers)

    def forward(self, x):
        x = self.transfer(x)
        return x
