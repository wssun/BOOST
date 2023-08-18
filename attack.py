import os
import sys

import torch.optim
from backdoors import *
from dataset import *
from helper import *
from util import *
import numpy as np
from torchvision.utils import save_image


class Attack:
    def __init__(self, model, args, device=None):
        self.device = device
        self.attack = args.attack
        self.target = args.target
        self.poison_rate = args.poison_rate

        self.shape = get_size(args.dataset)
        self.processing = get_norm(args.dataset)
        self.backdoor = get_backdoor(self.attack, self.shape,
                                     self.processing[0], self.device, args)

        self.opt_freq = 1
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(model.parameters(),
                                           lr=1e-2,
                                           momentum=0.9,
                                           weight_decay=5e-4)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                         step_size=50, gamma=0.1)

        self.train_set = get_dataset(args, train=True)
        self.test_set = get_dataset(args, train=False)
        self.poison_set = None

        if self.attack == 'badnets':
            pass

        elif self.attack == 'composite':
            CLASS_A = 0
            CLASS_B = 1
            CLASS_C = 2  # A + B -> C

            mixer = HalfMixer()

            self.train_set = MixDataset(dataset=self.train_set, mixer=mixer,
                                        classA=CLASS_A, classB=CLASS_B,
                                        classC=CLASS_C, data_rate=1,
                                        normal_rate=1.0, mix_rate=0,
                                        poison_rate=self.poison_rate)
            self.poison_set = MixDataset(dataset=self.test_set, mixer=mixer,
                                         classA=CLASS_A, classB=CLASS_B,
                                         classC=CLASS_C, data_rate=1,
                                         normal_rate=0, mix_rate=0,
                                         poison_rate=1)

            self.opt_freq = 2
            self.criterion = CompositeLoss(rules=[(CLASS_A, CLASS_B, CLASS_C)],
                                           simi_factor=1, mode='contrastive',
                                           device=self.device)
            self.optimizer = torch.optim.Adam(model.parameters())
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                             step_size=10, gamma=0.5)

        elif self.attack == 'wanet':
            self.noise_ratio = 2
            self.transform = PostTensorTransform(self.shape).to(self.device)

            self.train_set = get_dataset(args, train=True, augment=False)

            self.optimizer = torch.optim.SGD(model.parameters(), 1e-2,
                                             momentum=0.9, weight_decay=5e-4)
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, [100, 200, 300, 400], 0.1)
        elif self.attack == 'dynamic':
            self.optim_genr = torch.optim.Adam(
                self.backdoor.net_genr.parameters())
            self.sched_genr = torch.optim.lr_scheduler.MultiStepLR(
                self.optim_genr, [200, 300, 400, 500], 0.1)
        elif 'dfst' in self.attack:
            self.optim_genr_a2b = torch.optim.Adam(
                self.backdoor.genr_a2b.parameters(),
                2e-4, betas=(0.5, 0.999))
            self.optim_genr_b2a = torch.optim.Adam(
                self.backdoor.genr_b2a.parameters(),
                2e-4, betas=(0.5, 0.999))
            self.optim_disc_a = torch.optim.Adam(
                self.backdoor.disc_a.parameters(),
                2e-4, betas=(0.5, 0.999))
            self.optim_disc_b = torch.optim.Adam(
                self.backdoor.disc_b.parameters(),
                2e-4, betas=(0.5, 0.999))
        else:
            threat = 'dirty'
            if 'refool' in self.attack or self.attack == 'sig':
                threat = 'clean'

            self.train_set = PoisonDataset_npy(dataset=self.train_set,
                                           threat=threat, attack=self.attack,
                                           target=self.target, data_rate=1,
                                           processing=self.processing,
                                           poison_rate=self.poison_rate,
                                           backdoor=self.backdoor)

        if self.poison_set is None:
            self.poison_set = PoisonDataset(dataset=self.test_set, data_rate=1,
                                            threat='dirty', attack=self.attack,
                                            target=self.target, poison_rate=1,
                                            processing=self.processing,
                                            backdoor=self.backdoor)

    def inject(self, inputs, labels):
        if self.attack == 'badnets':
            normalize = self.processing[0]
            num_bd = int(inputs.size(0) * self.poison_rate)
            # print(num_bd)
            inputs_bd = self.backdoor.inject(inputs[:num_bd])
            labels_bd = torch.full((num_bd,), self.target).to(self.device)
            inputs = torch.cat([inputs_bd, inputs[num_bd:]], dim=0)
            labels = torch.cat([labels_bd, labels[num_bd:]], dim=0)
            inputs = normalize(inputs)
        elif self.attack == 'wanet':
            num_bd = int(inputs.size(0) * self.poison_rate)
            num_ns = int(num_bd * self.noise_ratio)

            inputs_bd = self.backdoor.inject(inputs[:num_bd])
            inputs_ns = self.backdoor.inject_noise(
                inputs[num_bd: (num_bd + num_ns)])

            labels_bd = torch.full((num_bd,), self.target).to(self.device)

            inputs = self.transform(torch.cat([inputs_bd, inputs_ns,
                                               inputs[(num_bd + num_ns):]], dim=0))
            labels = torch.cat([labels_bd, labels[num_bd:]], dim=0)
        elif self.attack in ['dynamic', 'dfst']:
            num_bd = int(inputs.size(0) * self.poison_rate)
            inputs_bd = self.backdoor.inject(inputs[:num_bd])
            labels_bd = torch.full((num_bd,), self.target).to(self.device)
            inputs = torch.cat([inputs_bd, inputs[num_bd:]], dim=0)
            labels = torch.cat([labels_bd, labels[num_bd:]], dim=0)

        return inputs, labels

class Attack_npy:
    def __init__(self, model, args, device=None):
        self.device = device
        self.attack = args.attack
        self.target = args.target
        self.poison_rate = args.poison_rate

        self.shape = get_size(args.dataset)
        self.processing = get_norm(args.dataset)
        self.backdoor = get_backdoor(self.attack, self.shape,
                                     self.processing[0], self.device, args)

        self.opt_freq = 1
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(model.parameters(),
                                            lr=1e-1,
                                            momentum=0.9,
                                            weight_decay=5e-4)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                         step_size=50, gamma=0.1)

        self.train_set = get_dataset(args, train=True)
        self.test_set = get_dataset(args, train=False)
        self.poison_set = None

        if self.attack == 'badnets':
            pass

        elif self.attack == 'composite':
            CLASS_A = 0
            CLASS_B = 1
            CLASS_C = 2  # A + B -> C

            mixer = HalfMixer()

            self.train_set = MixDataset_npy(dataset=self.train_set, mixer=mixer,
                                        classA=CLASS_A, classB=CLASS_B,
                                        classC=CLASS_C, data_rate=1,
                                        normal_rate=1.0, mix_rate=0,
                                        poison_rate=self.poison_rate)
            self.poison_set = MixDataset(dataset=self.test_set, mixer=mixer,
                                         classA=CLASS_A, classB=CLASS_B,
                                         classC=CLASS_C, data_rate=1,
                                         normal_rate=0, mix_rate=0,
                                         poison_rate=1)

            self.opt_freq = 2
            self.criterion = CompositeLoss(rules=[(CLASS_A, CLASS_B, CLASS_C)],
                                           simi_factor=1, mode='contrastive',
                                           device=self.device)
            self.optimizer = torch.optim.Adam(model.parameters())
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                             step_size=10, gamma=0.5)

        elif self.attack == 'wanet':
            self.noise_ratio = 2
            self.transform = PostTensorTransform(self.shape).to(self.device)

            self.train_set = get_dataset(args, train=True, augment=False)

            self.optimizer = torch.optim.SGD(model.parameters(), 1e-2,
                                             momentum=0.9, weight_decay=5e-4)
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, [100, 200, 300, 400], 0.1)
        elif self.attack == 'dynamic':
            self.optim_genr = torch.optim.Adam(
                self.backdoor.net_genr.parameters())
            self.sched_genr = torch.optim.lr_scheduler.MultiStepLR(
                self.optim_genr, [200, 300, 400, 500], 0.1)
        elif 'dfst' in self.attack:
            self.optim_genr_a2b = torch.optim.Adam(
                self.backdoor.genr_a2b.parameters(),
                2e-4, betas=(0.5, 0.999))
            self.optim_genr_b2a = torch.optim.Adam(
                self.backdoor.genr_b2a.parameters(),
                2e-4, betas=(0.5, 0.999))
            self.optim_disc_a = torch.optim.Adam(
                self.backdoor.disc_a.parameters(),
                2e-4, betas=(0.5, 0.999))
            self.optim_disc_b = torch.optim.Adam(
                self.backdoor.disc_b.parameters(),
                2e-4, betas=(0.5, 0.999))

        else:
            threat = 'dirty'

            if 'refool' in self.attack or self.attack == 'sig':
                threat = 'clean'

            self.train_set = PoisonDataset_npy(dataset=self.train_set,
                                           threat=threat, attack=self.attack,
                                           target=self.target, data_rate=1,
                                           processing=self.processing,
                                           poison_rate=self.poison_rate,
                                           backdoor=self.backdoor)

        if self.poison_set is None:
            self.poison_set = PoisonDataset(dataset=self.test_set, data_rate=1,
                                            threat='dirty', attack=self.attack,
                                            target=self.target, poison_rate=1,
                                            processing=self.processing,
                                            backdoor=self.backdoor)


    def inject(self, inputs, labels):
        if self.attack == "badnets":
            normalize = self.processing[0]

            num_bd = int(inputs.size(0) * self.poison_rate)
            inputs_bd = self.backdoor.inject(inputs[:num_bd])
            labels_bd = torch.full((num_bd,), self.target).to(self.device)

            poison = np.ones(num_bd, dtype=int)
            clean = np.zeros(inputs[num_bd:].size(0), dtype=int)

            inputs = (
            normalize(torch.cat([inputs_bd, inputs[num_bd:]], dim=0)), np.concatenate((poison, clean), axis=0))
            labels = torch.cat([labels_bd, labels[num_bd:]], dim=0)

        elif self.attack in ['dynamic', 'dfst']:
            num_bd = int(inputs.size(0) * self.poison_rate)
            inputs_bd = self.backdoor.inject(inputs[:num_bd])
            labels_bd = torch.full((num_bd,), self.target).to(self.device)

            poison = np.ones(num_bd, dtype=int)
            clean = np.zeros(inputs[num_bd:].size(0), dtype=int)

            inputs = (torch.cat([inputs_bd, inputs[num_bd:]], dim=0), np.concatenate((poison, clean), axis=0))
            labels = torch.cat([labels_bd, labels[num_bd:]], dim=0)

        elif self.attack == 'wanet':
            normalize = self.processing[0]

            num_bd = int(inputs.size(0) * self.poison_rate)
            num_ns = int(num_bd * self.noise_ratio)

            inputs_bd = self.backdoor.inject(inputs[:num_bd])
            inputs_ns = self.backdoor.inject_noise(
                inputs[num_bd: (num_bd + num_ns)])

            poison = np.ones(num_bd, dtype=int)
            clean = np.zeros(inputs[num_bd:].size(0), dtype=int)

            labels_bd = torch.full((num_bd,), self.target).to(self.device)

            inputs = self.transform(torch.cat([inputs_bd, inputs_ns,
                                               inputs[(num_bd + num_ns):]], dim=0))
            inputs = (inputs, np.concatenate((poison, clean), axis=0))
            labels = torch.cat([labels_bd, labels[num_bd:]], dim=0)
            
        return inputs, labels
