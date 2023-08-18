# coding: utf-8
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import argparse
import numpy as np
import os
import sys
import time
import torch

from torch.utils.data import DataLoader
from torchvision.utils import save_image
from attack import Attack, Attack_npy
from dataset import *
from helper import *
from inversion import *
from util import *
from torch.nn import Linear
import torch.nn

from config import attack_params
from pgd import PGD

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_DEVICE_ORDER']    = 'PCI_BUS_ID'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

_dataset_name = ['default', 'cifar10', 'gtsrb', 'imagenet', 'svhn', 'stl10']

_mean = {
    'default': [0.5, 0.5, 0.5],
    'cifar10': [0.4914, 0.4822, 0.4465],
    'gtsrb': [0.3337, 0.3064, 0.3171],
    'imagenet': [0.485, 0.456, 0.406],
    'svhn': [0.4377, 0.4438, 0.4728],
    'stl10': [0.4409, 0.4279, 0.3868],
}

_std = {
    'default': [0.5, 0.5, 0.5],
    'cifar10': [0.2023, 0.1994, 0.2010],
    'gtsrb': [0.2672, 0.2564, 0.2629],
    'imagenet': [0.229, 0.224, 0.225],
    'svhn': [0.1981, 0.2011, 0.1971],
    'stl10': [0.2551, 0.2480, 0.2564],
}

_size = {
    'cifar10': (32, 32),
    'gtsrb': (32, 32),
    'imagenet': (224, 224),
    'svhn': (32, 32),
    'stl10': (32, 32),
}

_num = {
    'cifar10': 10,
    'gtsrb': 43,
    'imagenet': 1000,
    'svhn': 10,
    'svhn': 10,
}
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

dataset_iso_model = r"./ckpt/PR0.1/wanet"

def get_norm(dataset):
    assert dataset in _dataset_name, _dataset_name
    mean = torch.FloatTensor(_mean[dataset])
    std = torch.FloatTensor(_std[dataset])
    normalize = transforms.Normalize(mean, std)
    unnormalize = transforms.Normalize(- mean / std, 1 / std)
    return normalize, unnormalize


def EuclideanDistance(x1,x2):
    temp = 0
    for i in range(x1.shape[0]):
        t1 = x1[i]
        t2 = x2[i]
        my_list = [x - y for x, y in zip(t1.tolist(), t2.tolist())]
        result = sum([x**2 for x in my_list])
        result = temp + result
    return result



def filter(tensor):
    random_mean0 = random.random()
    random_std0 = random.random()
    random_mean1 = random.random()
    random_std1 = random.random()
    random_mean2 = random.random()
    random_std2 = random.random()
    num = tensor.size()[0]
    x = torch.chunk(tensor, num, dim=0)
    for a in range(num):
        y = x[a].squeeze().cpu().detach().numpy()
        mean0_ori = y[0, :, :].mean()
        mean1_ori = y[1, :, :].mean()
        mean2_ori = y[2, :, :].mean()
        std0_ori = y[0, :, :].std()
        std1_ori = y[1, :, :].std()
        std2_ori = y[2, :, :].std()

        y[0, :, :] = (y[0, :, :] - mean0_ori) / std0_ori * random_std0 + random_mean0
        y[1, :, :] = (y[1, :, :] - mean1_ori) / std1_ori * random_std1 + random_mean1
        y[2, :, :] = (y[2, :, :] - mean2_ori) / std2_ori * random_std2 + random_mean2

        temp = torch.tensor(y)
        if a==0: new_img = temp.unsqueeze(0)
        else:
            new_img = torch.cat((new_img, temp.unsqueeze(0)), 0)

    return new_img


def cross(inputs, backdoor):
    size = inputs.size(0) // 2
    x_left  = backdoor.inject_noise(inputs[:size], inputs[size:])
    x_right = backdoor.inject_noise(inputs[size:], inputs[:size])
    inputs = torch.cat([x_left, x_right], dim=0)
    return inputs


def eval_acc(model, loader, backdoor=None):
    model.eval()
    n_sample = 0
    n_correct = 0
    with torch.no_grad():
        for step, (x_batch, y_batch) in enumerate(loader):
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)

            # x_batch = filter(x_batch)
            x_batch = x_batch.cuda()

            if backdoor is not None:
                x_batch = cross(x_batch, backdoor)

            output = model(x_batch)
            pred = output.max(dim=1)[1]

            n_sample  += x_batch.size(0)
            n_correct += (pred == y_batch).sum().item()

    acc = n_correct / n_sample
    return acc

def eval_acc_loss(model, loader, backdoor=None):
    model.eval()
    n_sample = 0
    n_correct = 0
    ori_sum = 0
    filter_sum = 0
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for step, (x_batch, y_batch) in enumerate(loader):
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)

            x_batch_ori = x_batch
            x_batch = filter(x_batch)
            x_batch = x_batch.cuda()

            output = model(x_batch)
            loss = criterion(output, y_batch)
            pred = output.max(dim=1)[1]
            # 对比ori和filter的平均loss值在前10个epoch中
            output_ori = model(x_batch_ori)
            loss_ori = criterion(output_ori, y_batch)

            if step < 10:
                ori_sum += loss_ori.item()
                filter_sum += loss.item()

            if backdoor is not None:
                x_batch = cross(x_batch, backdoor)
            n_sample  += x_batch.size(0)
            n_correct += (pred == y_batch).sum().item()

    acc = n_correct / n_sample
    return acc, ori_sum/10, filter_sum/10

def poison(args):
    model = get_model(args).to(DEVICE)
    model = torch.nn.DataParallel(model)

    attack = Attack(model, args, device=DEVICE)


    workers = 0 if args.attack == 'invisible' else 4

    train_loader  = DataLoader(dataset=attack.train_set,
                               batch_size=args.batch_size, shuffle=True)
    poison_loader = DataLoader(dataset=attack.poison_set,
                               batch_size=args.batch_size)
    test_loader   = DataLoader(dataset=attack.test_set,
                               batch_size=args.batch_size)

    save_path = f'ckpt/{args.dataset}_{args.network}_{args.attack}.pt'

    if args.attack == 'dfst':
        train_gan(attack, train_loader)
        torch.save(attack.backdoor.genr_a2b, f'{save_path[:-3]}_generator.pt')

    best_acc = 0
    best_asr = 0
    time_start = time.time()
    for epoch in range(args.epochs):
        model.train()
        if args.attack in ['dynamic']:
            attack.backdoor.net_genr.train()

        for step, (x_batch, y_batch) in enumerate(train_loader):
            if ('refool' in args.attack) or (args.attack == 'sig') or (args.attack == 'blend'):
                x_batch = x_batch[0].to(DEVICE)
                y_batch = y_batch.to(DEVICE)
                x_batch, y_batch = attack.inject(x_batch, y_batch)
            else:
                x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)

                x_batch, y_batch = attack.inject(x_batch, y_batch)

            attack.optimizer.zero_grad()
            if args.attack == 'dynamic':
                attack.optim_genr.zero_grad()

            output = model(x_batch)


            loss = attack.criterion(output, y_batch)

            loss.backward()
            attack.optimizer.step()
            if args.attack in ['dynamic']:
                attack.optim_genr.step()

            pred = output.max(dim=1)[1]
            acc = (pred == y_batch).sum().item() / x_batch.size(0)

            if step % 10 == 0:
                sys.stdout.write('\repoch {:3}, step: {:4}, loss: {:.4f}, '
                                 .format(epoch, step, loss) +\
                                 'acc: {:.4f}'.format(acc))
                sys.stdout.flush()

        attack.scheduler.step()
        if args.attack in ['dynamic']:
            attack.sched_genr.step()
            attack.backdoor.net_genr.eval()

        time_end = time.time()
        acc = eval_acc(model, test_loader)
        asr = eval_acc(model, poison_loader)

        sys.stdout.write('\repoch {:3}, step: {:4} - {:5.2f}s, acc: {:.4f}, '
                         .format(epoch, step, time_end-time_start, acc) +\
                         'asr: {:.4f}\n'.format(asr))
        sys.stdout.flush()
        time_start = time.time()

        # if epoch > 10 and acc + asr > best_acc + best_asr:
        if epoch < 20:
            best_acc = acc
            best_asr = asr
            print(f'---BEST ACC: {best_acc:.4f}, ASR: {best_asr:.4f}---')
            save_path = f'ckpt/PR0.1/refool_198/{args.attack}_{epoch}.pt'
            torch.save(model, save_path)
            if args.attack in ['dynamic']:
                torch.save(attack.backdoor.net_genr,
                           f'{save_path[:-3]}_pattern.pt')


def contrastive_loss(features, labels):
    features = features.view(features.shape[0], -1)
    features_0, features_1 = torch.chunk(features, 2)
    labels_0,   labels_1   = torch.chunk(labels,   2)

    # discard the main diagonal from both: labels and features matrix
    mask = torch.eye(labels_0.shape[0], dtype=torch.bool)
    labels = torch.matmul(labels_0, torch.t(labels_1))
    labels = labels[~mask].view(-1)
    features = torch.matmul(features_0, torch.t(features_1))
    features = features[~mask].view(-1)

    # select and combine multiple positives
    positive = torch.mean(features[labels.bool()])

    # select only the negatives
    negative = torch.mean(features[~labels.bool()])

    return positive - 0.1 * negative


###############################################################################
############                          main                         ############
###############################################################################
def main():
    if args.phase == 'poison':
        poison(args)
    else:
        print('Option [{}] is not supported!'.format(args.phase))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process input arguments.')

    parser.add_argument('--datadir', default='./data/cifar',    help='root directory of data')
    parser.add_argument('--suffix',  default='tmp',       help='suffix of saved path')
    parser.add_argument('--gpu',     default='0',         help='gpu id')

    parser.add_argument('--phase',   default='poison',      help='phase of framework')
    parser.add_argument('--dataset', default='cifar10',   help='dataset')
    parser.add_argument('--network', default='resnet18',     help='network structure')

    parser.add_argument('--attack',  default='dynamic',   help='attack type')
    parser.add_argument('--threat',  default='universal', help='threat model')
    parser.add_argument('--pair',    default='1-0',       help='label pair')

    parser.add_argument('--load',    action='store_true', help='load generated trigger')

    parser.add_argument('--seed',        type=int, default=1024, help='seed index')
    parser.add_argument('--batch_size',  type=int, default=128,  help='attack size')
    parser.add_argument('--epochs',      type=int, default=100,  help='number of epochs')
    parser.add_argument('--target',      type=int, default=0,    help='target label')

    parser.add_argument('--poison_rate', type=float, default=0.1,  help='poisoning rate')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    DEVICE = torch.device('cuda')

    time_start = time.time()
    main()
    time_end = time.time()
    print('='*50)
    print('Running time:', (time_end - time_start) / 60, 'm')
    print('='*50)
