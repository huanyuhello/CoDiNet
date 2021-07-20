
import torch
import torch.nn as nn
from torch.nn.modules.conv import _ConvNd

multiply_adds = 1


def count_conv2d(input, output, kernel_size, bias=None):
    batch_size = input.size()[0]
    out_h = output.size(2)
    out_w = output.size(3)
    cout = output.size()[1]
    cin = input.size()[1]
    kernel_ops = multiply_adds * kernel_size * kernel_size
    bias_ops = 1 if bias is not None else 0
    ops_per_element = kernel_ops + bias_ops

    output_elements = batch_size * out_w * out_h * cout
    total_ops = output_elements * ops_per_element * cin

    return total_ops


def count_bn(input):
    x = input[0]

    nelements = x.numel()
    # subtract, divide, gamma, beta
    total_ops = 4 * nelements

    return total_ops


def count_relu(input):
    x = input[0]

    nelements = x.numel()

    return nelements


def count_softmax(input):
    x = input[0]

    batch_size, nfeatures = x.size()

    total_exp = nfeatures
    total_add = nfeatures - 1
    total_div = nfeatures
    total_ops = batch_size * (total_exp + total_add + total_div)

    return total_ops


def count_avgpool(input, output, kernel_size):
    total_add = torch.prod(torch.tensor([kernel_size], device=input.device))
    total_div = 1
    kernel_ops = total_add + total_div
    num_elements = output.numel()
    total_ops = kernel_ops * num_elements

    return total_ops


def count_linear(input, c_in, c_out):
    total_mul = c_in
    total_add = c_in - 1
    total_ops = (total_mul + total_add) * c_out

    return total_ops
