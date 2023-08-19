import argparse
import os
import numpy as np
import torch
from openpyxl import Workbook
from openpyxl.styles import Font
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import sys
from attack import Attack_npy
from config import attack_params
from dataset import Dataset_npy
from pgd import PGD
from tqdm import tqdm
# Disable Warning
import warnings
import logging
import re
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
logging.captureWarnings(True)
warnings.filterwarnings('always', category=DeprecationWarning,
                        module=r'^{0}.'.format(re.escape(__name__)))
warnings.warn("This is a DeprecationWarning",category=DeprecationWarning)

sys.path.append("../..")

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

def get_norm(dataset):
    assert dataset in _dataset_name, _dataset_name
    mean = torch.FloatTensor(_mean[dataset])
    std = torch.FloatTensor(_std[dataset])
    normalize = transforms.Normalize(mean, std)
    unnormalize = transforms.Normalize(- mean / std, 1 / std)
    return normalize, unnormalize

def com_unex(list):
    cnt = 0
    Q1 = np.percentile(list, (25), interpolation='midpoint')
    Q3 = np.percentile(list, (75), interpolation='midpoint')
    Standard = Q3 + 1.5 * (Q3 - Q1)
    for idx in range(len(list)):
        if Standard < list[idx]:
            cnt+=1
    return cnt

def combine_dataset(data1, data2, path):
    combine = []
    for step, (x_batch, y_batch, z_batch) in enumerate(data1):
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
        for idx in range(x_batch.shape[0]):
            combine.append((x_batch[idx].cpu().numpy(), y_batch[idx].cpu().numpy(), z_batch[idx]))
    for step, (x_batch, y_batch, z_batch) in enumerate(data2):
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
        for idx in range(x_batch.shape[0]):
            combine.append((x_batch[idx].cpu().numpy(), y_batch[idx].cpu().numpy(), z_batch[idx]))
    np.save(path, combine)

def count_index2(Q1, dlist):
    cnt = []
    xx = dlist > Q1
    for i in range(dlist.size):
        if xx[0][i] == True: cnt.append(i)
    return cnt

def count_index(adv, truth):
    cnt = []
    for i in range(truth.size()[0]):
        xx = adv[i].item()
        yy = truth[i].item()
        if adv[i].item() == truth[i].item(): cnt.append(i)
    return cnt

def count(pre, turth):
    cnt = 0
    for i in range(turth.size()[0]):
        if pre[i] == turth[i]: cnt += 1
    return cnt

def readlist(tensor,list):
    for i in range(len(tensor)):
        list.append(tensor[i].item())
    return list

def read_directory(directory_name):
    Label = []
    for filename in tqdm(os.listdir(directory_name)):
        Label.append(filename)

    return Label

def select(args, model_name):
    save_path = f'./ckpt/PR0.1/refool_198/{model_name}.pt'
    normalize, unnormalize = get_norm(args.dataset)
    FT_model = torch.load(save_path).cuda()
    attack = Attack_npy(FT_model, args, device=DEVICE)
    train_loader = DataLoader(dataset=attack.train_set,
                              batch_size=args.batch_size, shuffle=True)

    flip_none = []
    flip = []
    cnt_flip = 0
    cnt_flipnone = 0
    for step, (x_batch, y_batch) in enumerate(train_loader):
        if args.attack in ['badnets', 'dfst', 'dynamic', 'inputaware', 'wanet']:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            x_batch, y_batch = attack.inject(x_batch, y_batch)
        elif args.attack in ['blend', 'sig', 'composite'] or 'refool' in args.attack:
            x_poison = x_batch[1].numpy()
            x_batch = (x_batch[0].to(DEVICE), x_poison)
            y_batch = y_batch.to(DEVICE)
            x_batch, y_batch = attack.inject(x_batch, y_batch)

        FT_model.eval()
        adversary = PGD(FT_model)
        AdvExArray = adversary.generate(unnormalize(x_batch[0]), y_batch, **attack_params['PGD_CIFAR10']).float()
        output = FT_model(x_batch[0])
        pred = output.max(dim=1)[1]
        adv_pred = FT_model(normalize(AdvExArray))
        adv_pred = adv_pred.argmax(dim=1, keepdim=True)

        index = count_index(adv_pred, y_batch)
        idx = 0
        if args.attack in ['dynamic', 'inputaware']:
            for idx in range(x_batch[0].size(0)):
                if idx in index:
                    flip_none.append((x_batch[0][idx].cpu().detach().numpy(), y_batch[idx].cpu().detach().numpy(), x_batch[1][idx]))
                    if x_batch[1][idx] == 1: cnt_flipnone += 1
                else:
                    flip.append((x_batch[0][idx].cpu().detach().numpy(), y_batch[idx].cpu().detach().numpy(), x_batch[1][idx]))
                    if x_batch[1][idx] == 1: cnt_flip += 1
        else:
            for idx in range(x_batch[0].size(0)):
                if idx in index:
                    flip_none.append((x_batch[0][idx].cpu().numpy(), y_batch[idx].cpu().numpy(), x_batch[1][idx]))
                    if x_batch[1][idx] == 1: cnt_flipnone += 1
                else:
                    flip.append((x_batch[0][idx].cpu().numpy(), y_batch[idx].cpu().numpy(), x_batch[1][idx]))
                    if x_batch[1][idx] == 1: cnt_flip += 1

    # print(len(flip), len(flip_none))

    FN_data_path = os.path.join("./iso_data/", args.attack + "_flip_none.npy")
    F_data_path = os.path.join("./iso_data/", args.attack+ "_flip.npy")
    np.save(FN_data_path, flip_none)
    np.save(F_data_path, flip)

    FT_data_path = os.path.join("./iso_data/", args.attack + "_flip.npy")
    FT_data = np.load(FT_data_path, allow_pickle=True)
    FT_dataX = Dataset_npy(full_dataset=FT_data, transform=True)
    FT_loader = DataLoader(dataset=FT_dataX,
                           batch_size=args.batch_size,
                           shuffle=True,
                           )
    F_data_path = os.path.join("./iso_data/", args.attack + "_flip_none.npy")
    F_data = np.load(F_data_path, allow_pickle=True)
    isolate_data = Dataset_npy(full_dataset=F_data, transform=True)
    train_loader = DataLoader(dataset=isolate_data,
                              batch_size=args.batch_size,
                              shuffle=True,
                              )

    combine_path = os.path.join("./iso_data/", args.attack + "_combine.npy")
    combine_dataset(FT_loader, train_loader, combine_path)
    combine_data = np.load(combine_path, allow_pickle=True)
    CB_data = Dataset_npy(full_dataset=combine_data, transform=True)
    combine_loader = DataLoader(dataset=CB_data,
                                batch_size=args.batch_size,
                                shuffle=True,
                                )

    xlist_total = []
    xlist_flip = []
    zlist_flip = []

    for step, (x_batch, y_batch, z_batch) in enumerate(FT_loader):
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)

        output = FT_model(x_batch)

        for idx in range(x_batch.size(0)):
            predict0 = output[idx]
            predict_max = predict0.max()
            xlist_flip.append(predict_max.item())
            zlist_flip.append(z_batch[idx].item())

    for step, (x_batch, y_batch, z_batch) in enumerate(combine_loader):
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)

        output = FT_model(x_batch)

        for idx in range(x_batch.size(0)):
            predict0 = output[idx]
            predict_max = predict0.max()
            xlist_total.append(predict_max.item())


    return xlist_total, xlist_flip, zlist_flip.count(1)



def main():
    best = 0
    best_rate = 1
    best_num = 50000
    for idx in range(20):
        # 批量
        name = args.attack + "_" + str(idx)
        total, flip, poi_num = select(args, name)
        flip_unex = com_unex(flip)
        total_unex = com_unex(total)
        diff = total_unex/len(total) - flip_unex/len(flip)
        rate = poi_num/len(flip)
        if diff > best:
            best = diff
            best_name1 = name
    print("best: ", best_name1, best)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process input arguments.')

    parser.add_argument('--datadir', default='./data/cifar', help='root directory of data')
    parser.add_argument('--suffix', default='tmp', help='suffix of saved path')
    parser.add_argument('--gpu', default='0', help='gpu id')

    parser.add_argument('--phase', default='poison', help='phase of framework')
    parser.add_argument('--dataset', default='cifar10', help='dataset')
    parser.add_argument('--network', default='resnet18', help='network structure')

    parser.add_argument('--attack', default='refool_smooth', help='attack type')
    parser.add_argument('--threat', default='universal', help='threat model')
    parser.add_argument('--pair', default='1-0', help='label pair')

    parser.add_argument('--load', action='store_true', help='load generated trigger')

    parser.add_argument('--seed', type=int, default=1024, help='seed index')
    parser.add_argument('--batch_size', type=int, default=128, help='attack size')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs')
    parser.add_argument('--target', type=int, default=0, help='target label')

    parser.add_argument('--poison_rate', type=float, default=0.1, help='poisoning rate')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    DEVICE = torch.device('cuda')
    main()




