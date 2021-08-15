# Code from https://github.com/simochen/model-tools.
import numpy as np
import argparse

import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable

import models

def print_model_param_nums(model=None, multiply_adds=True):
    if model == None:
        model = torchvision.models.alexnet()
    total = sum([param.nelement() for param in model.parameters()])
    # print('  + Number of params: %.2fM' % (total / 1e6))

    return total

def print_model_param_flops(model=None, input_res=224, multiply_adds=False, use_cuda=False):

    # prods = {}
    # def save_hook(name):
    #     def hook_per(self, input, output):
    #         prods[name] = np.prod(input[0].shape)
    #     return hook_per

    # list_1=[]
    # def simple_hook(self, input, output):
    #     list_1.append(np.prod(input[0].shape))
    # list_2={}
    # def simple_hook2(self, input, output):
    #     list_2['names'] = np.prod(input[0].shape)

    list_conv=[]
    def conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups)
        bias_ops = 1 if self.bias is not None else 0

        params = output_channels * (kernel_ops + bias_ops)
        flops = (kernel_ops * (2 if multiply_adds else 1) + bias_ops) * output_channels * output_height * output_width * batch_size

        list_conv.append(flops)

    list_linear=[]
    def linear_hook(self, input, output):
        batch_size = input[0].size(0) if input[0].dim() == 2 else 1

        weight_ops = self.weight.nelement() * (2 if multiply_adds else 1)
        bias_ops = self.bias.nelement()

        flops = batch_size * (weight_ops + bias_ops)
        list_linear.append(flops)

    list_bn=[]
    def bn_hook(self, input, output):
        list_bn.append(input[0].nelement() * 2)

    list_relu=[]
    def relu_hook(self, input, output):
        list_relu.append(input[0].nelement())

    list_pooling=[]
    def pooling_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size * self.kernel_size
        bias_ops = 0
        params = 0
        flops = (kernel_ops + bias_ops) * output_channels * output_height * output_width * batch_size

        list_pooling.append(flops)

    list_upsample=[]
    # For bilinear upsample
    def upsample_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        flops = output_height * output_width * output_channels * batch_size * 12
        list_upsample.append(flops)

    hooks = []
    def foo(net):
        childrens = list(net.children())
        if not childrens:
            if isinstance(net, torch.nn.Conv2d):
                handle = net.register_forward_hook(conv_hook)
                hooks.append(handle)
            if isinstance(net, torch.nn.Linear):
                handle = net.register_forward_hook(linear_hook)
                hooks.append(handle)
            if isinstance(net, torch.nn.BatchNorm2d):
                handle = net.register_forward_hook(bn_hook)
                hooks.append(handle)
            if isinstance(net, torch.nn.ReLU):
                handle = net.register_forward_hook(relu_hook)
                hooks.append(handle)
            if isinstance(net, torch.nn.MaxPool2d) or isinstance(net, torch.nn.AvgPool2d):
                handle = net.register_forward_hook(pooling_hook)
                hooks.append(handle)
            if isinstance(net, torch.nn.Upsample):
                handle = net.register_forward_hook(upsample_hook)
                hooks.append(handle)
            return
        for c in childrens:
            foo(c)

    # if model == None:
    #     model = torchvision.models.alexnet()
    total_flops = 0
    with torch.no_grad():
        foo(model)
        input = Variable(torch.rand(3, 3, input_res, input_res))
        if use_cuda:
            input = input.cuda()
        out = model(input)

        total_flops = (sum(list_conv) + sum(list_linear) + sum(list_bn) + sum(list_relu) + sum(list_pooling) + sum(list_upsample))
        # NOTE Remove all hooks to prevent memory leak
        for h in hooks:
            h.remove()
        # print('  + Number of FLOPs: %.5fM' % (total_flops / 3 / 1e6))
        return total_flops / 3

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Model param count')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='training dataset (default: cifar10)')
    parser.add_argument('--arch', default='vgg', type=str, 
                        help='architecture to use')
    parser.add_argument('--refine', default='', type=str, metavar='PATH',
                    help='path to the pruned model to be fine tuned')
    parser.add_argument('--depth', default=16, type=int,
                        help='depth of the neural network')

    args = parser.parse_args()

    model = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth)
    pruned = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth)
    if args.refine:
        checkpoint = torch.load(args.refine)
        pruned = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth, cfg=checkpoint['cfg'])
        pruned.load_state_dict(checkpoint['state_dict'])

    if args.dataset != "cifar10" and args.dataset != "cifar100":
        raise "Dataset need to be cifar10 or cifar100."
    
    print("\nDataset:", args.dataset, "Arch:", args.arch, "Depth: ", args.depth)

    # print("=" * 30, "\nThe origin model")
    model_flops = print_model_param_flops(model, 32)
    model_param = print_model_param_nums(model, 32)

    # print("=" * 30, "\nThe pruned model")
    pruned_model_flops = print_model_param_flops(pruned, 32)
    pruned_model_param = print_model_param_nums(pruned, 32)

    # print("=" * 30)
    print("\nParameters:", model_param, "->", pruned_model_param)
    print("FLOPS:", model_flops, "->", pruned_model_flops)
    print("\nPruned parameters:", round((model_param - pruned_model_param) / model_param, 4))
    print("Pruned FLOPS:", round((model_flops - pruned_model_flops) / model_flops, 4))
