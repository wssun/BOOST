import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
import os
from base_attack import BaseAttack

mean = [0.4914, 0.4822, 0.4465]
std = [0.2023, 0.1994, 0.2010]
normalize = transforms.Normalize(mean,std)
[0.4377, 0.4438, 0.4728]
[0.1981, 0.2011, 0.1971]
normalize2 = transforms.Normalize(mean,std)
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
class PGD(BaseAttack):
    """
    This is the multi-step version of FGSM attack.
    """


    def __init__(self, model, device = 'cuda'):

        super(PGD, self).__init__(model, device)

    def generate(self, image, label, **kwargs):
        """
        Call this function to generate PGD adversarial examples.

        Parameters
        ----------
        image :
            original image
        label :
            target label
        kwargs :
            user defined paremeters
        """

        ## check and parse parameters for attack
        label = label.type(torch.FloatTensor)

        assert self.check_type_device(image, label)
        assert self.parse_params(**kwargs)

        return pgd_attack(self.model,
                   self.image,
                   self.label,
                   self.epsilon,
                   self.clip_max,
                   self.clip_min,
                   self.num_steps,
                   self.step_size,
                   self.print_process,
                   self.bound,
                   self.attack_type)
                   ##default parameter for mnist data set.

    def parse_params(self,
                     epsilon = None,
                     num_steps = 1,
                     step_size = 0.0078,
                     clip_max = 1.0,
                     clip_min = 0.0,
                     print_process = False,
                     bound = 'linf',
                     attack_type = None
                     ):
        """parse_params.

        Parameters
        ----------
        epsilon :
            perturbation constraint
        num_steps :
            iteration step
        step_size :
            step size
        clip_max :
            maximum pixel value
        clip_min :
            minimum pixel value
        print_process :
            whether to print out the log during optimization process, True or False print out the log during optimization process, True or False.
        attack_type :
            backdoor attack
        """
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.clip_max = clip_max
        self.clip_min = clip_min
        self.print_process = print_process
        self.bound = bound
        self.attack_type = attack_type
        return True

def pgd_attack(model,
                  X,
                  y,
                  epsilon,
                  clip_max,
                  clip_min,
                  num_steps,
                  step_size,
                  print_process,
                  bound = 'linf',
                  attack_type = None):

    out = model(X)
    # err = (out.data.max(1)[1] != y.data).float().sum()
    #TODO: find a other way
    device = X.device
    imageArray = X.detach().cpu().numpy()
    X_random = np.random.uniform(-epsilon, epsilon, X.shape)
    imageArray = np.clip(imageArray + X_random, 0, 1.0)

    X_pgd = torch.tensor(imageArray).to(device).float()
    X_pgd.requires_grad = True
    eta = torch.zeros_like(X)
    eta.requires_grad = True

    # 统计在step中的loss变化
    loss_list = []

    for i in range(num_steps):
        if attack_type in ["DFST"]:
            pred = model(normalize2(X_pgd))
        else:
            pred = model(X_pgd)

        loss = nn.CrossEntropyLoss()(pred, y)
        loss_list.append(loss.item())

        if print_process:
            print("iteration {:.0f}, loss:{:.4f}".format(i,loss))

        loss.backward()

        if bound == 'linf':
            eta = step_size * X_pgd.grad.data.sign()
            X_pgd = X_pgd + eta
            eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)

            X_pgd = X.data + eta

            X_pgd = torch.clamp(X_pgd, clip_min, clip_max)
            #for ind in range(X_pgd.shape[1]):
            #    X_pgd[:,ind,:,:] = (torch.clamp(X_pgd[:,ind,:,:] * std[ind] + mean[ind], clip_min, clip_max) - mean[ind]) / std[ind]

            X_pgd = X_pgd.detach()
            X_pgd.requires_grad_()
            X_pgd.retain_grad()

        if bound == 'l2':
            output = model(X + eta)
            incorrect = output.max(1)[1] != y
            # correct = (~incorrect).unsqueeze(1).unsqueeze(1).unsqueeze(1).float()
            #Finding the correct examples so as to attack only them
            loss = nn.CrossEntropyLoss()(model(X + eta), y)
            loss.backward()

            eta.data +=  step_size * eta.grad.detach() / torch.norm(eta.grad.detach())
            eta.data *=  epsilon / torch.norm(eta.detach()).clamp(min=epsilon)
            eta.data =   torch.min(torch.max(eta.detach(), -X), 1-X) # clip X+delta to [0,1]
            eta.grad.zero_()
            X_pgd = X + eta

    return X_pgd

