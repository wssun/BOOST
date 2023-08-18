import numpy as np
import os
import torch
from backdoors import *
from models import *
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import torch.utils.data as data

EPSILON = 1e-7

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


def get_resize(size):
    if isinstance(size, str):
        assert size in _dataset_name, _dataset_name
        size = _size[size]
    return transforms.Resize(size)


def get_processing(dataset, augment=True, tensor=False, size=None):
    normalize, unnormalize = get_norm(dataset)

    transforms_list = []
    if size is not None:
        transforms_list.append(get_resize(size))
    if augment:
        transforms_list.append(transforms.RandomCrop(_size[dataset], padding=4))
        transforms_list.append(transforms.RandomHorizontalFlip())
    if not tensor:
        transforms_list.append(transforms.ToTensor())
    transforms_list.append(normalize)

    preprocess = transforms.Compose(transforms_list)
    deprocess = transforms.Compose([unnormalize])
    return preprocess, deprocess


def get_dataset(args, train=True, augment=True):
    transform, _ = get_processing(args.dataset, train & augment)
    transform_val = transforms.Compose([transforms.ToTensor()])
    if args.dataset == 'cifar10':
        if args.attack == 'badnets' and train == True:
            dataset = datasets.CIFAR10(args.datadir, train, transform_val, download=False)
        else:
            dataset = datasets.CIFAR10(args.datadir, train, transform, download=False)
    elif args.dataset == 'svhn':
        split = 'train' if train else 'test'
        if args.attack == 'badnets' and train == True:
            dataset = datasets.SVHN(args.datadir, split, transform_val, download=False)
        else:
            dataset = datasets.SVHN(args.datadir, split, transform, download=False)
        # dataset = datasets.SVHN(args.datadir, split, transform, download=False)
    elif args.dataset == 'stl10':
        transform, _ = get_processing(args.dataset, train & augment, size=[32,32])
        split = 'train' if train else 'test'
        dataset = datasets.STL10(args.datadir, split, transform=transform, download=False)
    elif args.dataset == 'gtsrb':
        transform, _ = get_processing(args.dataset, train & augment, size=[32,32])
        split = 1 if train else 0
        if args.attack == 'badnets' and split == 1:
            dataset = GTSRB(args.datadir, split, transform=transform_val)
        else:
            dataset = GTSRB(args.datadir, split, transform=transform)
    return dataset


def get_loader(args, train=True):
    dataset = get_dataset(args, train)
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size,
                            num_workers=4, shuffle=train)
    return dataloader


def get_model(args, pretrained=False):
    if args.network == 'resnet18':
        model = resnet18(args)
    elif args.network == 'preresnet18':
        model = preresnet18()
    elif args.network == 'vgg11':
        model = models.vgg11()
    return model


def get_classes(dataset):
    return _num[dataset]


def get_size(dataset):
    return _size[dataset]


def get_backdoor(attack, shape, normalize=None, device=None, args=None):
    if args is not None:
        base_path = f'ckpt/{args.dataset}_{args.network}'
    else:
        base_path = ''

    if 'refool' in attack:
        backdoor = Refool(shape, attack.split('_')[1], device=device)
    elif attack == 'wanet':
        backdoor = WaNet(shape, device=device)
        noise_path = f'{base_path}_wanet_noise.pt'
        identity_path = f'{base_path}_wanet_identity.pt'
        if os.path.exists(noise_path) & os.path.exists(identity_path):
            print("wanet.pt exsit!")
            backdoor.noise_grid = torch.load(noise_path).to(device)
            backdoor.identity_grid = torch.load(identity_path).to(device)
        else:
            torch.save(backdoor.noise_grid.cpu(), noise_path)
            torch.save(backdoor.identity_grid.cpu(), identity_path)
    elif attack == 'invisible':
        backdoor = Invisible()
    elif attack in ['blend', 'sig', 'polygon']:
        backdoor = Other(attack, device=None)
    elif attack == 'filter':
        backdoor = Filter()
    elif attack == 'badnets':
        backdoor = Badnets(normalize, device=None)
    elif attack == 'inputaware':
        backdoor = InputAware(normalize, device=device)
        mask_path = f'{base_path}_inputaware_mask.pt'
        genr_path = f'{base_path}_inputaware_pattern.pt'
        if os.path.exists(mask_path) & os.path.exists(genr_path):
            print("inputaware.pt exsit!")
            backdoor.net_mask = torch.load(mask_path).to(device)
            backdoor.net_genr = torch.load(genr_path).to(device)
    elif attack == 'dynamic':
        backdoor = Dynamic(normalize, device=device)
        genr_path = f'{base_path}_dynamic_pattern.pt'
        if os.path.exists(genr_path):
            print("dynamic_pattern.pt exsit!")
            backdoor.net_genr = torch.load(genr_path).to(device)
    elif 'dfst' in attack:
        backdoor = DFST(normalize, device=device)
        genr_path = f'{base_path}_dfst_generator.pt'
        if os.path.exists(genr_path):
            print("dfst_generator.pt exsit!")
            backdoor.genr_a2b = torch.load(genr_path).to(device)
    else:
        backdoor = None
    return backdoor

class GTSRB(data.Dataset):
    def __init__(self, data_root, train, transform):
        super(GTSRB, self).__init__()
        self.classes = ['Speed limit 20km/h',
                        'Speed limit 30km/h',
                        'Speed limit 50km/h',
                        'Speed limit 60km/h',
                        'Speed limit 70km/h',
                        'Speed limit 80km/h',  # 5
                        'End of speed limit 80km/h',
                        'Speed limit 100km/h',
                        'Speed limit 120km/h',
                        'No passing sign',
                        'No passing for vehicles over 3.5 metric tons',  # 10
                        'Right-of-way at the next intersection',
                        'Priority road sign',
                        'Yield sign',
                        'Stop sign',  # 14
                        'No vehicles sign',  # 15
                        'Vehicles over 3.5 metric tons prohibited',
                        'No entry',
                        'General caution',
                        'Dangerous curve to the left',
                        'Dangerous curve to the right',  # 20
                        'Double curve',
                        'Bumpy road',
                        'Slippery road',
                        'Road narrows on the right',
                        'Road work',  # 25
                        'Traffic signals',
                        'Pedestrians crossing',
                        'Children crossing',
                        'Bicycles crossing',
                        'Beware of ice or snow',  # 30
                        'Wild animals crossing',
                        'End of all speed and passing limits',
                        'Turn right ahead',
                        'Turn left ahead',
                        'Ahead only',  # 35
                        'Go straight or right',
                        'Go straight or left',
                        'Keep right',
                        'Keep left',
                        'Roundabout mandatory',  # 40
                        'End of no passing',
                        'End of no passing by vehicles over 3.5 metric tons']

        if train:
            self.input_array = np.load(f'{data_root}/train.npz')
        else:
            self.input_array = np.load(f'{data_root}/test.npz')
        self.data = self.input_array['x']
        self.targets = self.input_array['y'][:, 0].tolist()
        self.transforms = transform
        self.root = data_root

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transforms is not None:
            img = self.transforms(img)
        return img, target
