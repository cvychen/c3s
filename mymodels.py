import math
# import os
import torch
import torch.nn as nn
from torchvision.models import ResNet
# import numpy as np
# import torch.nn.functional as F
# import pdb

# eps = np.finfo(float).eps

# os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


##################################### Loss functions
class RegularLoss(nn.Module):

    def __init__(self, gamma=0, part_features=None, nparts=1):
        """
        :param bs: batch size
        :param ncrops: number of crops used at constructing dataset
        """
        super(RegularLoss, self).__init__()
        self.register_buffer('part_features', part_features)
        self.nparts = nparts
        self.gamma = torch.Tensor([gamma])
        # self.batchsize = bs
        # self.ncrops = ncrops

    def forward(self, x):
        assert isinstance(x, list), "parts features should be presented in a list"
        corr_matrix = torch.zeros(self.nparts, self.nparts).cuda()
        loss = torch.zeros(1, requires_grad=True).cuda()
        # x = [torch.div(xx, xx.norm(dim=1, keepdim=True)) for xx in x]
        for i in range(self.nparts):
            x[i] = x[i].squeeze()
            # x[i] = x[i].view(self.batchsize, self.ncrops, -1).mean(1)
            x[i] = torch.div(x[i], x[i].norm(dim=1, keepdim=True))
        for i in range(self.nparts):
            for j in range(self.nparts):
                corr_matrix[i, j] = torch.mean(torch.mm(x[i], x[j].t()))
                if i == j:
                    corr_matrix[i, j] = torch.abs(corr_matrix[i, j] - 1.0)

        loss = torch.sum(torch.triu(corr_matrix))

        return torch.mul(self.gamma.cuda(), loss)


#class Hiera_RegularLoss(nn.Module):
#    def __init__(self, gamma=0, logits=None, class_distance=None):
#        super(Hiera_RegularLoss, self).__init__()
#        self.register_buffer('logits', logits)
#        self.class_distance = torch.Tensor(class_distance)
#        self.gamma = torch.Tensor([gamma])
#
#    def forward(self, x, label):
#        batch_size = x.size(0)
#        distance = self.class_distance[label].cuda()
#        reg_loss = torch.zeros(1, requires_grad=True).cuda()
#        reg_loss = torch.mm(x.view(1, -1), distance.view(-1, 1))
#
#        return torch.mul(self.gamma.cuda(), reg_loss) / batch_size

class Hiera_RegularLoss(nn.Module):
    def __init__(self, gamma=0, logits=None, class_distance=None):
        super(Hiera_RegularLoss, self).__init__()
        self.register_buffer('logits', logits)
        self.class_distance = torch.Tensor(class_distance)
        self.gamma = torch.Tensor([gamma])

    def forward(self, x, label):
        batch_size = x.size(0)
        distance = self.class_distance[label].cuda()
        reg_loss = torch.zeros(1, requires_grad=True).cuda()
        reg_loss = torch.sum(torch.mul(x, distance))

        return torch.mul(reg_loss, self.gamma.cuda()) / batch_size

##################################### Squeeze-and-Excitation modules
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class MELayer(nn.Module):
    def __init__(self, channel, reduction=16, nparts=1):
        super(MELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.nparts = nparts
        parts = list()
        for part in range(self.nparts):
            parts.append(nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
            ))
        self.parts = nn.Sequential(*parts)
        self.dresponse = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)

        meouts = list()
        for i in range(self.nparts):
            meouts.append(x * self.parts[i](y).view(b, c, 1, 1))

        y = self.dresponse(y).view(b, c, 1, 1)
        return x * y, meouts

##################################### ResBlocks
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, nparts=1, meflag=False, reduction=16):
        super(ResBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.meflag = meflag
        if self.meflag:
            self.se = MELayer(planes * 4, reduction=reduction, nparts=nparts)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if self.meflag:
            parts = self.se(out)
            for i in range(len(parts)):
                parts[i] = self.relu(parts[i] + residual)
            return parts
        else:
            out += residual
            out = self.relu(out)
            return out

##################################### SEBlocks
class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SEBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16, meflag=False, nparts=1):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.meflag = meflag
        if self.meflag:
            self.se = MELayer(planes * 4, reduction=reduction, nparts=nparts)
        else:
            self.se = SELayer(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        #pdb.set_trace()
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if self.meflag:
            out, parts = self.se(out)

            out += residual
            out = self.relu(out)

            for i in range(len(parts)):
                parts[i] = self.relu(parts[i] + residual)

            return out, parts
        else:
            out = self.se(out)
            out += residual
            out = self.relu(out)
            return out

###################################### ResNet framework
class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, rd=[16, 16, 16, 16], nparts=1, seflag=False):
        """
        :param rd: reductions in SENet
        :param seflag: Ture for SENet, Flase for ResNet

        """
        super(ResNet, self).__init__()

        self.inplanes = 64
        self.seflag = seflag
        self.rd = rd
        self.nparts = nparts

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], reduction=self.rd[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, reduction=self.rd[1])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, reduction=self.rd[2], meflag=False, nparts=nparts)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, reduction=self.rd[3], meflag=True, nparts=nparts)
        self.adpavgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion * nparts, num_classes)

        # initializing params
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, reduction=16, meflag=False, nparts=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, reduction))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            if i == blocks - 1 and meflag is True:
                layers.append(block(self.inplanes, planes, reduction=reduction, meflag=meflag, nparts=nparts))
            else:
                layers.append(block(self.inplanes, planes, reduction=reduction))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        _, ftrs = self.layer4(x)

        cmbn_ftres = list()
        for i in range(self.nparts):
            ftrs[i] = self.adpavgpool(ftrs[i])

        # for the final layer
        xf = torch.cat(ftrs, 1)
        xf = xf.view(xf.size(0), -1)
        xf = self.fc(xf)
        return xf, ftrs



########################################## Models
def feasc18(num_classes=200, nparts=1):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBasicBlock, [2, 2, 2, 2], num_classes=num_classes, nparts=nparts)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def feasc34(num_classes=200, nparts=1):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBasicBlock, [3, 4, 6, 3], num_classes=num_classes, nparts=nparts)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def feasc50(num_classes=200, nparts=1, seflag=False):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if seflag:
        rd = [16, 32, 64, 128]
        model = ResNet(SEBottleneck, [3, 4, 6, 3], num_classes=num_classes, rd=rd, nparts=nparts, seflag=True)
    else:
        model = ResNet(ResBottleneck, [3, 4, 6, 3], num_classes=num_classes, nparts=nparts, seflag=False)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def feasc101(num_classes=200, nparts=1):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBottleneck, [3, 4, 23, 3], num_classes=num_classes, nparts=nparts)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def feasc152(num_classes=200, nparts=1):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBottleneck, [3, 8, 36, 3], num_classes=num_classes, nparts=nparts)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model
