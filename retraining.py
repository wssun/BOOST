# coding: utf-8
import matplotlib.pyplot as plt
# Disable Warning
import warnings
import logging
import re
import copy
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
logging.captureWarnings(True)
warnings.filterwarnings('always', category=DeprecationWarning,
                        module=r'^{0}.'.format(re.escape(__name__)))
warnings.warn("This is a DeprecationWarning",category=DeprecationWarning)

import pandas as pd
from config import attack_params
from pgd import PGD

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import argparse
import time
from attack import Attack, Attack_npy
from dataset import *
from inversion import *
from util import *

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

dataset_iso_model = r"./ckpt"

def get_norm(dataset):
    assert dataset in _dataset_name, _dataset_name
    mean = torch.FloatTensor(_mean[dataset])
    std = torch.FloatTensor(_std[dataset])
    normalize = transforms.Normalize(mean, std)
    unnormalize = transforms.Normalize(- mean / std, 1 / std)
    return normalize, unnormalize



def writetxt(txtpath, num1_poi, num1_cle, num2_poi, num2_cle, flag):
    if flag == True:
        file = open(txtpath, 'w+')
        file.write('Total fliped: {}, poison: {}, clean: {}'.format(num1_poi + num1_cle, num1_poi, num1_cle))
        file.write('\n')
        file.write('Total none fliped: {}, poison: {}, clean: {}'.format(num2_poi + num2_cle, num2_poi, num2_cle))
    else:
        file = open(txtpath, 'w+')
        file.write('Total right: {}, poison: {}, clean: {}'.format(num1_poi + num1_cle, num1_poi, num1_cle))
        file.write('\n')
        file.write('Total false: {}, poison: {}, clean: {}'.format(num2_poi + num2_cle, num2_poi, num2_cle))

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


def eval_acc(model, loader, backdoor=None):
    model.eval()
    n_sample = 0
    n_correct = 0
    with torch.no_grad():
        for step, (x_batch, y_batch) in enumerate(loader):
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)

            x_batch = x_batch.cuda()

            if backdoor is not None:
                x_batch = cross(x_batch, backdoor)

            output = model(x_batch)
            pred = output.max(dim=1)[1]

            n_sample += x_batch.size(0)
            n_correct += (pred == y_batch).sum().item()

    acc = n_correct / n_sample
    return acc


def cross(inputs, backdoor):
    size = inputs.size(0) // 2
    x_left = backdoor.inject_noise(inputs[:size], inputs[size:])
    x_right = backdoor.inject_noise(inputs[size:], inputs[:size])
    inputs = torch.cat([x_left, x_right], dim=0)
    return inputs


def adv(args):
    normalize, unnormalize = get_norm(args.dataset)
    name = args.model_name + '.pt'
    model = torch.load(os.path.join(dataset_iso_model, name)).to(DEVICE)
    # model = torch.nn.DataParallel(model)
    print("\nUsing the {} model\n".format(args.model_name))
    print("Poison rate is {}\n".format(args.poison_rate))
    attack = Attack_npy(model, args, device=DEVICE)

    train_loader = DataLoader(dataset=attack.train_set,
                              batch_size=args.batch_size, shuffle=True)

    if args.attack == 'dfst':
        pass

    time_start = time.time()
    flip_none = []
    flip = []
    cnt_flip = 0
    cnt_flipnone = 0
    for step, (x_batch, y_batch) in enumerate(train_loader):
        if args.attack in ['badnets', 'dfst', 'dynamic', 'wanet']:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            x_batch, y_batch = attack.inject(x_batch, y_batch)
            x_batch_true = copy.copy(x_batch)
            y_batch_true = copy.copy(y_batch)
        elif args.attack in ['blend', 'sig', 'composite'] or 'refool' in args.attack:
            x_poison = x_batch[1].numpy()
            x_batch = (x_batch[0].to(DEVICE), x_poison)
            y_batch = y_batch.to(DEVICE)
            x_batch, y_batch = attack.inject(x_batch, y_batch)
            x_batch_true = copy.copy(x_batch)
            y_batch_true = copy.copy(y_batch)


        model.eval()
        adversary = PGD(model)
        AdvExArray = adversary.generate(unnormalize(x_batch[0]), y_batch, **attack_params['PGD_CIFAR10']).float()

        output = model(x_batch[0])
        pred = output.max(dim=1)[1]

        adv_pred = model(normalize(AdvExArray))
        adv_pred = adv_pred.argmax(dim=1, keepdim=True)

        acc = (pred == y_batch).sum().item() / x_batch[0].size(0)
        num = count(adv_pred.cpu(), y_batch.cpu())
        adv = 1 - count(adv_pred.cpu(), y_batch.cpu()) / x_batch[0].size(0)

        index = count_index(adv_pred, y_batch)
        idx = 0
        if args.attack in ['dynamic', 'inputaware']:
            for idx in range(x_batch[0].size(0)):
                if idx in index:
                    flip_none.append((x_batch_true[0][idx].cpu().detach().numpy(), y_batch_true[idx].cpu().detach().numpy(), x_batch_true[1][idx]))
                    if x_batch[1][idx] == 1: cnt_flipnone += 1
                else:
                    flip.append((x_batch_true[0][idx].cpu().detach().numpy(), y_batch_true[idx].cpu().detach().numpy(), x_batch_true[1][idx]))
                    if x_batch[1][idx] == 1: cnt_flip += 1
        elif args.attack == 'wanet':
            for idx in range(x_batch[0].size(0)):
                if idx in index:
                    flip_none.append((normalize(AdvExArray[idx]).cpu().detach().numpy(), y_batch_true[idx].cpu().detach().numpy(), x_batch_true[1][idx]))
                    if x_batch[1][idx] == 1: cnt_flipnone += 1
                else:
                    flip.append((normalize(AdvExArray[idx]).cpu().detach().numpy(), y_batch_true[idx].cpu().detach().numpy(), x_batch_true[1][idx]))
                    if x_batch[1][idx] == 1: cnt_flip += 1
        else:
            for idx in range(x_batch[0].size(0)):
                if idx in index:
                    flip_none.append((x_batch_true[0][idx].cpu().numpy(), y_batch_true[idx].cpu().numpy(), x_batch_true[1][idx]))
                    if x_batch[1][idx] == 1: cnt_flipnone += 1
                else:
                    flip.append((x_batch_true[0][idx].cpu().numpy(), y_batch_true[idx].cpu().numpy(), x_batch_true[1][idx]))
                    if x_batch[1][idx] == 1: cnt_flip += 1


    time_end = time.time()
    sys.stdout.write('Finish adv attack! - {:5.2f}s,\n'.format(time_end - time_start))
    sys.stdout.flush()
    time_start = time.time()

    print("Total {} data are flipped, there is {} poison data and {} clean data".format(len(flip), cnt_flip,
                                                                                        len(flip) - cnt_flip))
    print(
        "Total {} data are not flipped, there is {} poison data and {} clean data".format(len(flip_none), cnt_flipnone,
                                                                                          len(flip_none) - cnt_flipnone))
    txtpath = './logs/{}_filp_result.txt'.format(args.model_name)
    writetxt(txtpath, cnt_flip, len(flip) - cnt_flip, cnt_flipnone, len(flip_none) - cnt_flipnone, True)
    FN_data_path = os.path.join("./iso_data/", args.model_name + "_flip_none.npy")
    F_data_path = os.path.join("./iso_data/", args.model_name + "_flip.npy")
    np.save(FN_data_path, flip_none)
    np.save(F_data_path, flip)


def poison(args):
    model = get_model(args).to(DEVICE)

    attack = Attack_npy(model, args, device=DEVICE)

    F_data_path = os.path.join("./iso_data/", args.model_name + "_flip.npy")

    isolate_poisoned_data = np.load(F_data_path, allow_pickle=True)

    poisoned_data_tf = Dataset_npy(full_dataset=isolate_poisoned_data, transform=True)
    train_loader = DataLoader(dataset=poisoned_data_tf,
                              batch_size=args.batch_size,
                              shuffle=True,
                              )
    print("Load the {}\n".format(F_data_path))
    poison_loader = DataLoader(dataset=attack.poison_set,
                               batch_size=args.batch_size)
    test_loader = DataLoader(dataset=attack.test_set,
                             batch_size=args.batch_size)

    if args.attack == 'dfst':
        pass

    best_acc = 0
    best_asr = 0
    time_start = time.time()
    fineturn_process = []
    for epoch in range(args.epochs):
        model.train()
        for step, (x_batch, y_batch, __) in enumerate(train_loader):
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)

            attack.optimizer.zero_grad()

            output = model(x_batch)
            loss = attack.criterion(output, y_batch)
            loss.backward()
            attack.optimizer.step()

            pred = output.max(dim=1)[1]
            acc = (pred == y_batch).sum().item() / x_batch.size(0)

            if step % 10 == 0:
                sys.stdout.write('\repoch {:3}, step: {:4}, loss: {:.4f}, '
                                 .format(epoch, step, loss) + \
                                 'acc: {:.4f}\n'.format(acc))
                sys.stdout.flush()

        attack.scheduler.step()

        time_end = time.time()
        acc = eval_acc(model, test_loader)
        asr = eval_acc(model, poison_loader)

        sys.stdout.write('\repoch {:3}, step: {:4} - {:5.2f}s, acc: {:.4f}, '
                         .format(epoch, step, time_end - time_start, acc) + \
                         'asr: {:.4f}\n'.format(asr))
        sys.stdout.flush()
        time_start = time.time()

        fineturn_process.append((args.dataset, args.batch_size, args.poison_rate, epoch, acc, asr, loss.item()))
        df = pd.DataFrame(fineturn_process,
                          columns=("dataname-", "-batch_size-", "-poison_rate-", "-epoch-", "-Acc-", "-Asr-", "-loss-"))
        df.to_csv("./logs/{} with {}.csv".format(args.model_name, args.poison_rate), index=False, encoding="utf-8", header=True)

        save_path = f'ckpt/{args.model_name}_init.pt'

        if epoch > 10 and acc > best_acc:
            best_acc = acc
            best_asr = asr
            print(f'---BEST ACC: {best_acc:.4f}, ASR: {best_asr:.4f}---')
            torch.save(model, save_path)

    txtpath = f'ckpt/{args.model_name}_to_{best_acc:.4f}_{best_asr:.4f}_record.txt'
    file = open(txtpath, 'w+')
    file.write('\n')

def pre(args, i, save_path):
    if i == 0: save_path = f'ckpt/{args.model_name}_init.pt'
    model = torch.load(save_path).cuda()
    FN_data_path = os.path.join("./iso_data/", args.model_name + "_flip_none.npy")
    isolate_data = np.load(FN_data_path, allow_pickle=True)
    data_tf = Dataset_npy(full_dataset=isolate_data, transform=True)
    train_loader = DataLoader(dataset=data_tf,
                              batch_size=args.batch_size,
                              shuffle=True,
                              )
    print("Load the {}\n".format(FN_data_path))

    pre_true = []
    pre_false = []
    prelist = []
    cnt_right = 0
    cnt_false = 0
    for step, (x_batch, y_batch, z_batch) in enumerate(train_loader):
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)

        output = model(x_batch)
        pred = output.max(dim=1)[1]

        index = count_index(pred, y_batch)
        for idx in range(x_batch.size(0)):
            if idx in index:
                pre_true.append((x_batch[idx].cpu().numpy(), y_batch[idx].cpu().numpy(), z_batch[idx]))
                if z_batch[idx] == 1: cnt_right += 1
                # 计算四分位数
                predict0 = output[idx]
                predict_max = predict0.max()
                prelist.append(predict_max.item())
            else:
                pre_false.append((x_batch[idx].cpu().numpy(), y_batch[idx].cpu().numpy(), z_batch[idx]))
                if z_batch[idx] == 1: cnt_false += 1

    print("Total {} data are right predicted, there is {} poison data and {} clean data".format(len(pre_true),
                                                                                                cnt_right,
                                                                                                len(pre_true) - cnt_right))
    print(
        "Total {} data are not right predicted, there is {} poison data and {} clean data".format(len(pre_false),
                                                                                                  cnt_false,
                                                                                                  len(pre_false) - cnt_false))
    Q1 = np.percentile(prelist, (25), interpolation='midpoint')
    txtpath = './logs/{}_pred_result_{}.txt'.format(args.model_name, i)
    writetxt(txtpath, cnt_right, len(pre_true) - cnt_right, cnt_false, len(pre_false) - cnt_false, False)
    FT_data_path = os.path.join("./iso_data/", args.model_name + "_finetune.npy")
    np.save(FT_data_path, pre_true)

    print("Reload FT_dataset")
    FT_data = np.load(FT_data_path, allow_pickle=True)
    FT_data = Dataset_npy(full_dataset=FT_data, transform=True)
    FT_loader = DataLoader(dataset=FT_data,
                           batch_size=args.batch_size,
                           shuffle=True,
                           )
    Q_true = []
    cntt = 0
    for step, (x_batch, y_batch, z_batch) in enumerate(FT_loader):
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
        output = model(x_batch)
        for j in range(x_batch.size(0)):
            predict0 = output[j]
            predict_max = predict0.max()
            if predict_max > Q1:
                Q_true.append((x_batch[j].cpu().numpy(), y_batch[j].cpu().numpy(), z_batch[j]))
                if z_batch[j] == 1: cntt += 1
    print("Total {} data are selected, there is {} poison data and {} clean data".format(len(Q_true), cntt,
                                                                                         len(Q_true) - cntt))
    txtpath = './logs/{}_Q1_result_{}.txt'.format(args.model_name, i)
    writetxt(txtpath, cntt, len(Q_true) - cntt, cntt, len(Q_true) - cntt, False)
    Q1_data_path = os.path.join("./iso_data/", args.model_name + "_Q1.npy")
    np.save(Q1_data_path, Q_true)

def finetune(args, i):
    save_path = f'ckpt/{args.model_name}_init.pt'
    FT_model = torch.load(save_path).cuda()
    attack = Attack_npy(FT_model, args, device=DEVICE)

    Q1_data_path = os.path.join("./iso_data/", args.model_name + "_Q1.npy")
    Q1_data = np.load(Q1_data_path, allow_pickle=True)
    Q1_data = Dataset_npy(full_dataset=Q1_data, transform=True)
    Q1_loader = DataLoader(dataset=Q1_data,
                           batch_size=args.batch_size,
                           shuffle=True
                           )
    F_data_path = os.path.join("./iso_data/", args.model_name + "_flip.npy")
    F_data = np.load(F_data_path, allow_pickle=True)
    isolate_data = Dataset_npy(full_dataset=F_data, transform=True)
    train_loader = DataLoader(dataset=isolate_data,
                              batch_size=args.batch_size,
                              shuffle=True,
                              )

    combine_path = os.path.join("./iso_data/", args.model_name + "_combine.npy")
    combine_dataset(Q1_loader, train_loader, combine_path)
    combine_data = np.load(combine_path, allow_pickle=True)
    CB_data = Dataset_npy(full_dataset=combine_data, transform=True)
    combine_loader = DataLoader(dataset=CB_data,
                              batch_size=args.batch_size,
                              shuffle=True,
                              )

    print("Load the {}\n".format(combine_path))
    poison_loader = DataLoader(dataset=attack.poison_set,
                               batch_size=args.batch_size)
    test_loader = DataLoader(dataset=attack.test_set,
                             batch_size=args.batch_size)

    time_start = time.time()
    best_acc = 0
    best_asr = 0
    fineturn_process = []
    for epoch in range(args.FT_epochs):
        FT_model.train()
        for step, (x_batch, y_batch, __) in enumerate(combine_loader):
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)

            attack.optimizer.zero_grad()
            output = FT_model(x_batch)
            loss = attack.criterion(output, y_batch)
            loss.backward()
            attack.optimizer.step()

            pred = output.max(dim=1)[1]
            acc = (pred == y_batch).sum().item() / x_batch.size(0)


        attack.scheduler.step()

        time_end = time.time()
        acc = eval_acc(FT_model, test_loader)
        asr = eval_acc(FT_model, poison_loader)

        sys.stdout.write('\repoch {:3}, step: {:4} - {:5.2f}s, acc: {:.4f}, '
                         .format(epoch, step, time_end - time_start, acc) + \
                         'asr: {:.4f}\n'.format(asr))
        sys.stdout.flush()
        time_start = time.time()

        fineturn_process.append((args.dataset, args.batch_size, args.poison_rate, epoch, acc, asr, loss.item()))
        df = pd.DataFrame(fineturn_process,
                          columns=(
                              "dataname-", "-batch_size-", "-poison_rate-", "-epoch-", "-Acc-", "-Asr-", "-loss"))
        df.to_csv("./logs/{} with {}_finetune_{}.csv".format(args.model_name, args.poison_rate, i), index=False,
                  encoding="utf-8",
                  header=True)

        if acc > best_acc:
            best_acc = acc
            best_asr = asr
            best_model = FT_model
            print(f'---BEST ACC: {best_acc:.4f}, ASR: {best_asr:.4f}---')

    save_path = f'ckpt/{args.model_name}_to_{best_acc:.4f}_{best_asr:.4f}_ft_{i}.pt'
    torch.save(best_model, save_path)
    return save_path



###############################################################################
############                          main                         ############
###############################################################################
def main():
    adv(args)
    poison(args)
    save_path = ''
    for i in range(4):
        pre(args, i, save_path)
        save_path = finetune(args, i)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process input arguments.')

    parser.add_argument('--datadir', default='./data/cifar', help='root directory of data')
    parser.add_argument('--suffix', default='tmp', help='suffix of saved path')
    parser.add_argument('--gpu', default='0', help='gpu id')
    parser.add_argument('--model_name', type=str, default='dfst_10', help='adv the model')

    parser.add_argument('--phase', default='poison', help='phase of framework')
    parser.add_argument('--dataset', default='cifar10', help='dataset')
    parser.add_argument('--network', default='resnet18', help='network structure')

    parser.add_argument('--attack', default='dfst', help='attack type')
    parser.add_argument('--threat', default='universal', help='threat model')
    parser.add_argument('--pair', default='1-0', help='label pair')

    parser.add_argument('--load', action='store_true', help='load generated trigger')

    parser.add_argument('--seed', type=int, default=1024, help='seed index')
    parser.add_argument('--batch_size', type=int, default=100, help='attack size')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--target', type=int, default=0, help='target label')

    parser.add_argument('--poison_rate', type=float, default=0.1, help='poisoning rate')
    parser.add_argument('--FT_epochs', type=int, default=100, help='number of finetune epochs')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    DEVICE = torch.device('cuda')

    time_start = time.time()
    main()
    time_end = time.time()
    print('=' * 50)
    print('Running time:', (time_end - time_start) / 60, 'm')
    print('=' * 50)
