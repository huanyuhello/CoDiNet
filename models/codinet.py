'''
Properly implemented ResNet-s for CIFAR10 as described in paper [1].
The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.
Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:
name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m
which this implementation indeed has.
'''

import torch.nn.functional as F
import torch.nn.init as init

from utils.ops import *
from utils.utils import get_blocks

def _weights_init(m):
    # classname = m.__class__.__name__
    # print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


############################################################################################################
# this is the basic block
############################################################################################################

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU()
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant",
                                                  0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, padding=0, bias=False),
                    nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu2(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

############################################################################################################
# Gates
############################################################################################################

class Gate(nn.Module):
    def __init__(self, in_planes, options, beta=0, freeze_gate=False, hard_sample=False):
        super(Gate, self).__init__()
        self.hard_sample = hard_sample
        self.line0 = nn.Linear(in_planes, in_planes)
        self.line1 = nn.Linear(in_planes, options)

        if freeze_gate:
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, x):
        out = F.avg_pool2d(x, x.size(2))
        out = out.view(out.size(0), -1)

        out = self.line0(out)
        out = self.line1(out)

        out = F.gumbel_softmax(out, hard=self.hard_sample)

        out = out.view(out.size(0), out.size(1), 1, 1)
        return out

############################################################################################################
# Block with Gate
############################################################################################################

class GateBlock(nn.Module):
    # expansion = 1

    def __init__(self, Gate, in_planes, planes, stride=1, beta=25, finetune=False, blockType=BasicBlock,
                 freeze_gate=False, hard_sample=False):
        super(GateBlock, self).__init__()
        self.block = blockType(in_planes, planes, stride)

        self.gate = Gate(in_planes, options=2, beta=beta, freeze_gate=freeze_gate, hard_sample=hard_sample)

        self.finetune = finetune
        self.expansion = self.block.expansion
        if stride != 1 or in_planes != self.expansion * planes:
            self.skip = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
                nn.ReLU())
        else:
            self.skip = nn.Sequential()

    def forward(self, input):
        gate = self.gate(input)
        out0 = self.block(input)
        out1 = self.skip(input)
        if self.training and not self.finetune:
            mask0 = gate[:, 0, :, :].unsqueeze(1)
            mask1 = gate[:, 1, :, :].unsqueeze(1)
            out = out0 * mask0 + out1 * mask1
        else:
            _, mask = gate.max(1)
            mask0 = torch.where(mask == 0, torch.full_like(mask, 1), torch.full_like(mask, 0)).float()
            mask1 = torch.where(mask == 1, torch.full_like(mask, 1), torch.full_like(mask, 0)).float()
            out = out0 * mask0.unsqueeze(1) + out1 * mask1.unsqueeze(1)

        return out, gate


############################################################################################################
# Network
############################################################################################################
class CoDiNet_Cifar(nn.Module):
    def __init__(self, gate, block, num_blocks, num_classes=10, beta=25, finetune=False, freeze_gate=False,
                 hard_sample=False):
        super(CoDiNet_Cifar, self).__init__()
        self.in_planes = 16
        self.beta = beta
        self.gate = gate
        self.num_class = num_classes
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1, finetune=finetune, freeze_gate=freeze_gate,
                                       hard_sample=hard_sample)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2, finetune=finetune, freeze_gate=freeze_gate,
                                       hard_sample=hard_sample)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2, finetune=finetune, freeze_gate=freeze_gate,
                                       hard_sample=hard_sample)
        self.linear = nn.Linear(64, num_classes)
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride, finetune, freeze_gate, hard_sample):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = nn.ModuleList()
        for stride in strides:
            layers.append(
                block(self.gate, self.in_planes, planes, stride, beta=self.beta, finetune=finetune,
                      freeze_gate=freeze_gate, hard_sample=hard_sample))
            self.in_planes = planes * BasicBlock.expansion
        return layers

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        probs = []
        for block in self.layer1:
            outx, prob = block(out)
            out = outx
            probs.append(prob)
        for block in self.layer2:
            outx, prob = block(out)
            out = outx
            probs.append(prob)
        for block in self.layer3:
            outx, prob = block(out)
            out = outx
            probs.append(prob)

        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        gprob = torch.stack(probs, dim=0).permute(1, 0, 2, 3, 4)
        return out, gprob


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, beta=25):
        super(ResNet, self).__init__()
        self.in_planes = 16
        self.beta = beta
        self.conv1 = Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = BatchNorm2d(16)
        self.relu1 = ReLU()
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = Linear(64, num_classes)
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = nn.ModuleList()
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return layers

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        for block in self.layer1:
            out = block(out)
        for block in self.layer2:
            out = block(out)
        for block in self.layer3:
            out = block(out)

        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class CoDiNet_ImageNet(nn.Module):
    def __init__(self, gate, block, num_blocks, num_classes=10, beta=25, finetune=False, freeze_gate=False,
                 hard_sample=False):
        super(CoDiNet_ImageNet, self).__init__()

        self.in_planes = 64
        self.beta = beta
        self.gate = gate
        self.num_class = num_classes

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, finetune=finetune, freeze_gate=freeze_gate,
                                       hard_sample=hard_sample)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, finetune=finetune, freeze_gate=freeze_gate,
                                       hard_sample=hard_sample)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, finetune=finetune, freeze_gate=freeze_gate,
                                       hard_sample=hard_sample)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, finetune=finetune, freeze_gate=freeze_gate,
                                       hard_sample=hard_sample)
        self.linear = nn.Linear(512 * 4, num_classes)
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride, finetune, freeze_gate, hard_sample):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = nn.ModuleList()
        for stride in strides:
            layers.append(block(self.gate, self.in_planes, planes, stride, beta=self.beta, finetune=finetune,
                                blockType=Bottleneck, freeze_gate=freeze_gate, hard_sample=hard_sample))
            self.in_planes = planes * Bottleneck.expansion
        return layers

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        probs = []

        for block in self.layer1:
            out, prob = block(out)
            probs.append(prob)
        for block in self.layer2:
            out, prob = block(out)
            probs.append(prob)
        for block in self.layer3:
            out, prob = block(out)
            probs.append(prob)
        for block in self.layer4:
            out, prob = block(out)
            probs.append(prob)

        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        gprob = torch.stack(probs, dim=0).permute(1, 0, 2, 3, 4)

        return out, gprob


def CoDiNet(backbone, num_class, beta=25, finetune=False, freeze_gate=False, hard_sample=False):
    num_block = get_blocks(backbone)
    if isinstance(num_block, list):
        return CoDiNet_ImageNet(Gate, GateBlock, num_block, num_class, beta=beta, finetune=finetune,
                           freeze_gate=freeze_gate, hard_sample=hard_sample)
    else:
        return CoDiNet_Cifar(Gate, GateBlock, [num_block, num_block, num_block], num_class, beta=beta,
                       finetune=finetune, freeze_gate=freeze_gate, hard_sample=hard_sample)


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size()) > 1, net.parameters()))))
