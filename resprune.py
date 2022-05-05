import argparse
import numpy as np
import os
import time
import random
from scipy.spatial import distance

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms

from models import *
import algorithm
import compute_flops

from hyperopt import hp
from hyperopt import fmin, tpe, space_eval

# Prune settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR prune')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='training dataset (default: cifar10)')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--depth', type=int, default=56,
                    help='depth of the resnet')
parser.add_argument('--model', default='', type=str, metavar='PATH',
                    help='path to the model (default: none)')
parser.add_argument('--save', default='', type=str, metavar='PATH',
                    help='path to save pruned model (default: none)')
parser.add_argument('-v', default='A', type=str, 
                    help='version of the model')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--finetune-epoch', type=int, default=1, metavar='N',
                    help='how many epoch to finetune before evaluation')
# Algorithm
parser.add_argument('--algo',       default="sem",  type=str,    help='algorithm')
parser.add_argument('--pop',        default=30,     type=int,    help='population size')
parser.add_argument('--iter',       default=100,    type=int,    help='iterations')
parser.add_argument('--cr',         default=0.7,    type=float,  help='ga/de crossover rate')
parser.add_argument('--mr',         default=0.1,    type=float,  help='ga mutation rate')
parser.add_argument('--f',          default=0.4,    type=float,  help='de f')
parser.add_argument('--w',          default=1,    type=float,  help='pso w')
parser.add_argument('--c1',         default=2,    type=float,  help='pso c1')
parser.add_argument('--c2',         default=2,    type=float,  help='pso c2')
parser.add_argument('--region',     default=4,      type=int,    help='se region')
parser.add_argument('--searcher',   default=1,      type=int,    help='se searcher')
parser.add_argument('--sample',     default=4,      type=int,    help='se sample')
parser.add_argument('--player',     default=3,      type=int,    help='se player')
parser.add_argument('--cthre',      default=0.5,    type=float,  help='se crossover thre')
parser.add_argument('--mthre',      default=0.1,   type=float,  help='se mutation thre')
parser.add_argument('--tpe',        default=1000,   type=int,  help='tpe evaluations')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if not os.path.exists(args.save):
    os.makedirs(args.save)

# Build mdoel
model = resnet(depth=args.depth, dataset=args.dataset)
if args.cuda:
    model.cuda()

if args.model:
    if os.path.isfile(args.model):
        print("=> loading checkpoint '{}'".format(args.model))
        checkpoint = torch.load(args.model)
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
              .format(args.model, checkpoint['epoch'], best_prec1))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

# Load data
kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}

if args.dataset == 'mnist':
    full_train_data = datasets.MNIST('./data/mnist', download=True, train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])), 
elif args.dataset == 'cifar10':
    full_train_data = datasets.CIFAR10('./data/cifar10', train=True, transform=transforms.Compose([
        transforms.Pad(4),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))
elif args.dataset == 'cifar100':
    full_train_data = datasets.CIFAR100('./data/cifar100', train=True, transform=transforms.Compose([
        transforms.Pad(4),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))
elif args.dataset == 'svhn':
    full_train_data = datasets.SVHN('./data/svhn', split='train', download=True,
        transform=transforms.Compose([
        transforms.Pad(4),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
else:
    raise ValueError("No valid dataset is given.")
# Split validation data from train data
if args.dataset == 'mnist':
    train_data, valid_data = torch.utils.data.random_split(full_train_data, [50000,10000])
elif args.dataset == 'svhn':
    train_data, valid_data = torch.utils.data.random_split(full_train_data, [40000,33257])
else:
    train_data, valid_data = torch.utils.data.random_split(full_train_data, [40000,10000])
partial_train_data, _ = torch.utils.data.random_split(train_data, [10000,30000])
train_loader = torch.utils.data.DataLoader(partial_train_data, batch_size=args.batch_size, shuffle=True, **kwargs)
validation_loader = torch.utils.data.DataLoader(valid_data, batch_size=args.test_batch_size, shuffle=True, **kwargs)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
print('Pre-processing Successful!')

# simple test model after Pre-processing prune (simple set BN scales to zeros)
def test(model):
    with torch.no_grad():
        model.eval()
        correct = 0
        for data, target in validation_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        # print('\nTest set: Accuracy: {}/{} ({:.1f}%)\n'.format(
        #     correct, len(validation_loader.dataset), 100. * correct / len(validation_loader.dataset)))
    return correct / float(len(validation_loader.dataset))

def get_dim(model, is_binary=False):
    skip = {
        '56': {
            'A': [16, 20, 38, 54],
            'B': [16, 18, 20, 34, 38, 54],
        },
        '110': {
            'A': [36],
            'B': [36, 38, 74],
        },
    }
    layer_id = 1
    dim = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            if layer_id in skip[str(args.depth)][args.v]:
                layer_id += 1
                continue
            if layer_id % 2 == 0:
                if is_binary:
                    dim += m.weight.data.shape[0]
                else:
                    dim += 1
                layer_id += 1
                continue
            layer_id += 1

    return 2 * dim

def convert_solution(solution, is_binary=False):
    skip = {
        '56': {
            'A': [16, 20, 38, 54],
            'B': [16, 18, 20, 34, 38, 54],
        },
        '110': {
            'A': [36],
            'B': [36, 38, 74],
        },
    }
    layer_id = 1
    cfg = []
    cfg_mask = []
    s = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            out_channels = m.weight.data.shape[0]
            if layer_id in skip[str(args.depth)][args.v]:
                cfg_mask.append(torch.ones(out_channels))
                cfg.append(out_channels)
                layer_id += 1
                continue
            if layer_id % 2 == 0:
                new_out_channels = round(out_channels * solution[len(solution) // 2 + s])
                if new_out_channels < 1:
                    new_out_channels = 1
                if new_out_channels > out_channels:
                    new_out_channels = out_channels
                cfg.append(new_out_channels)

                sigmoid = 1 / (1 + np.exp(-solution[s]))
                if random.random() > sigmoid == 1:
                    cfg_mask.append(get_layer_mask_geometric(m, new_out_channels))
                else:
                    cfg_mask.append(get_layer_mask_norm(m, new_out_channels))

                s += 1
                layer_id += 1
                continue

            layer_id += 1

    return cfg, cfg_mask

def apply_weights(model, newmodel, cfg_mask):
    layer_id_in_cfg = 0
    conv_count = 1
    for [m0, m1] in zip(model.modules(), newmodel.modules()):
        if isinstance(m0, nn.Conv2d):
            if conv_count == 1:
                m1.weight.data = m0.weight.data.clone()
                conv_count += 1
                continue
            if conv_count % 2 == 0:
                mask = cfg_mask[layer_id_in_cfg]
                idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
                if idx.size == 1:
                    idx = np.resize(idx, (1,))
                w = m0.weight.data[idx.tolist(), :, :, :].clone()
                m1.weight.data = w.clone()
                layer_id_in_cfg += 1
                conv_count += 1
                continue
            if conv_count % 2 == 1:
                mask = cfg_mask[layer_id_in_cfg-1]
                idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
                if idx.size == 1:
                    idx = np.resize(idx, (1,))
                w = m0.weight.data[:, idx.tolist(), :, :].clone()
                m1.weight.data = w.clone()
                conv_count += 1
                continue
        elif isinstance(m0, nn.BatchNorm2d):
            if conv_count % 2 == 1:
                mask = cfg_mask[layer_id_in_cfg-1]
                idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
                if idx.size == 1:
                    idx = np.resize(idx, (1,))
                m1.weight.data = m0.weight.data[idx.tolist()].clone()
                m1.bias.data = m0.bias.data[idx.tolist()].clone()
                m1.running_mean = m0.running_mean[idx.tolist()].clone()
                m1.running_var = m0.running_var[idx.tolist()].clone()
                continue
            m1.weight.data = m0.weight.data.clone()
            m1.bias.data = m0.bias.data.clone()
            m1.running_mean = m0.running_mean.clone()
            m1.running_var = m0.running_var.clone()
        elif isinstance(m0, nn.Linear):
            m1.weight.data = m0.weight.data.clone()
            m1.bias.data = m0.bias.data.clone()

def get_pruned_ratio(model, new_model):
    total = sum([param.nelement() for param in model.parameters()])
    new_total = sum([param.nelement() for param in new_model.parameters()])

    return (total - new_total) / total

def get_pruned_flops_ratio(model, new_model):
    # NOTE Use input res 32 for cifar dataset
    model.cpu()
    new_model.cpu()
    total = compute_flops.print_model_param_flops(model=model, input_res=32, use_cuda=False)
    new_total = compute_flops.print_model_param_flops(model=new_model, input_res=32, use_cuda=False)
    if args.cuda:
        model.cuda()
        new_model.cuda()

    return (total - new_total) / total

def get_layer_mask_geometric(m, keep_num):
    mask = []
    out_channels = m.weight.data.shape[0]
    if out_channels == keep_num:               # No filter pruned in this layer
        mask = torch.ones(out_channels)
        return mask

    weight_torch = m.weight.data
    weight_vec = weight_torch.view(weight_torch.size()[0], -1)
    
    # TODO Remove norm based pruning, this code is for mixed version (l1+GM) from the FPGM paper
    # Reference: https://github.com/he-y/filter-pruning-geometric-median/issues/43
    norm = torch.norm(weight_vec, 2, 1)
    norm_np = norm.cpu().numpy()
    filter_pruned_num = 0
    filter_large_index = norm_np.argsort()[filter_pruned_num:]
    indices = torch.LongTensor(filter_large_index).cuda()
    weight_vec_after_norm = torch.index_select(weight_vec, 0, indices).cpu().numpy()

    similar_matrix = distance.cdist(weight_vec_after_norm, weight_vec_after_norm, 'euclidean')
    similar_sum = np.sum(np.abs(similar_matrix), axis=0)
    similar_pruned_num = out_channels - keep_num
    similar_small_index = similar_sum.argsort()[:  similar_pruned_num]
    similar_index_for_filter = [filter_large_index[i] for i in similar_small_index]

    assert len(similar_index_for_filter) == out_channels - keep_num, "size of similar_index_for_filter not correct"
    mask = torch.ones(out_channels)
    mask[similar_index_for_filter] = 0

    return mask

def get_layer_mask_norm(m, keep_num):
    # NOTE: Uncomment the code to use feature map rank for sorting
    # with open("rank.txt", "r") as f:
    #     ranks = f.read()
    # ranks = eval(ranks)
    # r_idx = 0

    mask = []
    out_channels = m.weight.data.shape[0]
    if out_channels == keep_num:                    # No filter pruned in this layer
        mask = torch.ones(out_channels)
        # r_idx += 1
        return mask
    # NOTE: L1-norm
    weight_copy = m.weight.data.abs().clone()       # Abs
    weight_copy = weight_copy.cpu().numpy()
    L1_norm = np.sum(weight_copy, axis=(1, 2, 3))   # Sum the entire filter
    arg_max = np.argsort(L1_norm)                   # Sort norm values

    # NOTE: Feature map rank
    # arg_max = np.argsort(ranks[r_idx])            # Sort rank values
    # r_idx += 1

    arg_max_rev = arg_max[::-1][:keep_num]          # Reverse arg_max and get the top portion we want
    assert arg_max_rev.size == keep_num, "size of arg_max_rev not correct"
    mask = torch.zeros(out_channels)
    mask[arg_max_rev.tolist()] = 1

    return mask

def finetune(model, curr_epoch):
    model.train()
    # avg_loss = 0.
    # train_acc = 0.
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        # avg_loss += loss.item()
        # pred = output.data.max(1, keepdim=True)[1]
        # train_acc += pred.eq(target.data.view_as(pred)).cpu().sum()
        loss.backward()
        optimizer.step()
    # print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(
    #     curr_epoch, batch_idx * len(data), len(train_loader.dataset),
    #     100. * batch_idx / len(train_loader), loss.item()))

def evaluate(solution, is_binary=False, save_model=False, save_name=None):
    cfg, cfg_mask = convert_solution(solution, is_binary)
    new_model = resnet(dataset=args.dataset, depth=args.depth, cfg=cfg)
    if args.cuda:
        new_model.cuda()
    
    # Apply weights from original model
    apply_weights(model, new_model, cfg_mask)

    # NOTE Finetune model with little data
    best_prec1 = 0
    for epoch in range(args.finetune_epoch): # finetune for a few epochs
        finetune(new_model, epoch + 1)
        prec1 = test(new_model)
        best_prec1 = max(prec1, best_prec1)

    # Evaluate model with pruned ratio and accuracy
    valid_acc_ratio = best_prec1.clone().item()
    pruned_ratio = get_pruned_flops_ratio(model, new_model) # NOTE Pruned FLOPS ratio
    # pruned_ratio = get_pruned_ratio(model, new_model)
    acc_drop_norm = ((origin_acc - valid_acc_ratio) / origin_acc).item()
    fitness = ((1 - pruned_ratio) + acc_drop_norm) / 2

    if save_model:
        print(new_model)
        tar_name = "pruned.pth.tar"
        txt_name = "prune.txt"
        if save_name:
            tar_name = "pruned_" + save_name + ".pth.tar"
            txt_name = "prune_" + save_name + ".txt"
        torch.save({'cfg': cfg, 'state_dict': new_model.state_dict()}, os.path.join(args.save, tar_name))
        num_parameters = sum([param.nelement() for param in new_model.parameters()])
        with open(os.path.join(args.save, txt_name), "w") as fp:
            fp.write("Number of parameters: \n"+str(num_parameters)+"\n")
            fp.write("Valid accuracy: \n"+str(valid_acc_ratio)+"\n")
    
    print("{} Pruned ratio: {}, Test Acc: {}, Fitness: {}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                                                            round(pruned_ratio, 4), 
                                                            round(valid_acc_ratio * 100, 2), 
                                                            round(fitness, 4)))

    return fitness, pruned_ratio, valid_acc_ratio


origin_acc = test(model)

# Run algorithm
use_binary = False
if args.algo in ["seb"]:
    use_binary = True

dim = get_dim(model, is_binary=use_binary)
best = [0 for _ in range(dim)]
ratio_list = []

if args.algo == "seb":
    print("Start searching with SE binary, region ", str(args.region), "searcher", str(args.searcher), "sample", str(args.sample))
    algo = algorithm.SEB(dim, args.pop, args.region, args.searcher, args.sample, args.cthre, args.mthre)
    best, ratio_list = algo.run(args.iter, evaluate_func=evaluate)
    print("SE binary search end, found best:")

if args.algo == "sem":
    print("Start searching with SE modified, region ", str(args.region), "searcher", str(args.searcher), "sample", str(args.sample))
    algo = algorithm.SEM(dim, args.pop, args.region, args.searcher, args.sample, args.player, args.cthre, args.mthre)
    best, ratio_list, best_fitness = algo.run(args.iter, evaluate_func=evaluate)
    print("SE modified search end, found best:")
    
if args.algo == "se":
    print("Start searching with SE, region ", str(args.region), "searcher", str(args.searcher), "sample", str(args.sample))
    algo = algorithm.SE(dim, args.pop, args.region, args.searcher, args.sample)
    best = algo.run(args.iter, evaluate_func=evaluate)
    print("SE search end, found best:")

if args.algo == "de":
    print("Start searching with DE, cr", str(args.cr), "f", str(args.f))
    algo = algorithm.DE(dim, args.pop, args.cr, args.f)
    best, ratio_list = algo.run(args.iter, evaluate_func=evaluate)
    print("DE search end, found best:")

if args.algo == "ga":
    print("Start searching with GA, cr", str(args.cr), "mr", str(args.mr))
    algo = algorithm.GA(dim, args.pop, args.cr, args.mr)
    best, ratio_list = algo.run(args.iter, evaluate_func=evaluate)
    print("GA search end, found best:")

if args.algo == "pso":
    print("Start searching with PSO, w", str(args.w), "c1", str(args.c1), "c2", str(args.c2))
    algo = algorithm.PSO(dim, args.pop, args.w, args.c1, args.c2)
    best, ratio_list = algo.run(args.iter, evaluate_func=evaluate)
    print("PSO search end, found best:")

if args.algo == "gwo":
    print("Start searching with GWO")
    algo = algorithm.GWO(dim, args.pop)
    best, ratio_list = algo.run(args.iter, evaluate_func=evaluate)
    print("GWO search end, found best:")

if args.algo == "random":
    print("Start searching with Random")
    algo = algorithm.Random(dim, args.pop)
    best, ratio_list = algo.run(args.pop * args.iter, evaluate_func=evaluate)
    print("Random search end, found best:")

# SE Parameter Search Using TPE
if args.algo == "tpe":
    def tpe_objective(tpe_args):
        algo = algorithm.SEM(dim, args.pop, tpe_args["region"], tpe_args["searcher"], tpe_args["sample"], tpe_args["player"], tpe_args["cthre"], tpe_args["mthre"])
        _, _, best_fitness = algo.run(1000000, evaluate_func=evaluate)
        print("TPE one evaluation done, fitness:", best_fitness)

        return best_fitness

    search_space = {
        'region'    : hp.choice('region', [1, 2, 3, 4, 5, 6, 7, 8]),
        'searcher'  : hp.choice('searcher', [1, 2, 3, 4, 5, 6, 7, 8]),
        'sample'    : hp.choice('sample', [1, 2, 3, 4, 5, 6, 7, 8]),
        'player'    : hp.choice('player', [1, 2, 3, 4, 5, 6, 7, 8]),
        'cthre'     : hp.uniform('cthre', 0, 1),
        'mthre'     : hp.uniform('mthre', 0, 1),
    }

    best = fmin(tpe_objective, search_space, algo=tpe.suggest, max_evals=args.tpe, verbose=False)
    print(best)
    # print(space_eval(space, best))


evaluate(best, save_model=True, save_name="best")
for sol in ratio_list:
    if not sol.fitness == float("inf"):
        evaluate(sol.position, save_model=True, save_name=str(round(sol.pruned, 4)))
