#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torchvision import datasets, transforms
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal, mnist_noniid_mix, mnist_noniid_transformed
from sampling import cifar_iid, cifar_noniid, cifar_noniid_transformed, cifar_noniid_mix, cifar_noniid_mix_original
from torch import nn
import torch.nn.functional as F
from torch.optim import Optimizer
import torch.nn.utils.parametrize as parametrize
from models import WeightAverageParametrization
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from torch.utils.data import random_split
import numpy as np




class TransformDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        # implement the train_labels to contain all labels of the dataset
        self.train_labels = []
        for i in range(len(dataset)):
            self.train_labels.append(dataset[i][1])
        self.train_labels = torch.tensor(self.train_labels)

    def __getitem__(self, index):
        x, y = self.dataset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.dataset)


def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    combine_weights_train_id = []
    
    if args.dataset == 'cifar' or args.dataset == 'cifar_transformed':
        if args.dataset == 'cifar':
            data_dir = '../data/cifar/'
        elif args.dataset == 'cifar_transformed':
            data_dir = '../data/cifar_transformed/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        if (args.dataset == 'cifar_transformed'):
            train_dataset = torch.load(data_dir + 'cifar_train_combined.pt')
            test_dataset = torch.load(data_dir + 'cifar_test_combined.pt')

            train_dataset = TransformDataset(train_dataset, transform=apply_transform)
            test_dataset = TransformDataset(test_dataset, transform=apply_transform)
        else:
            train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                        transform=apply_transform)
            

            test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                        transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args.num_users)
        elif not args.mix:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                user_groups, combine_weights_train_id = cifar_noniid(train_dataset, args.num_users)
        elif (args.mix == 1):
            print("mix case 1")
            if(args.dataset == 'cifar'):
                user_groups, combine_weights_train_id = cifar_noniid_mix_original(train_dataset, args.num_users)
            elif(args.dataset == 'cifar_transformed'):
                user_groups, combine_weights_train_id = cifar_noniid_mix(train_dataset, args.num_users)
        elif (args.mix == 2):
            print("mix case 2")
            user_groups, combine_weights_train_id = cifar_noniid_transformed(train_dataset, args.num_users)

    elif args.dataset == 'mnist' or 'fmnist' or 'mnist_transformed':
        if args.dataset == 'mnist':
            data_dir = '../data/mnist/'
        elif args.dataset == 'mnist_transformed':
            data_dir = '../data/mnist_transformed/'
        else:
            data_dir = '../data/fmnist/'

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])
        
        if(args.dataset == 'mnist_transformed'):
            train_dataset = torch.load(data_dir + 'mnist_train_combined.pt')
            test_dataset = torch.load(data_dir + 'mnist_test_combined.pt')

            train_dataset = TransformDataset(train_dataset, transform=apply_transform)
            test_dataset = TransformDataset(test_dataset, transform=apply_transform)

        else:

            train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                        transform=apply_transform)

            test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                        transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_users)
        elif not args.mix:
            print("not mixed")
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups, combine_weights_train_id = mnist_noniid(train_dataset, args.num_users)
        elif (args.mix == 1):
            print("mix case 1")
            user_groups, combine_weights_train_id = mnist_noniid_mix(train_dataset, args.num_users)
        elif (args.mix == 2):
            print("mix case 2")
            user_groups, combine_weights_train_id = mnist_noniid_transformed(train_dataset, args.num_users)

    return train_dataset, test_dataset, user_groups, combine_weights_train_id






def model_register_parametrization(model, input_models, alpha):
    for name, module in model.named_modules():
        if(isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d)):
            parametrize.register_parametrization(module, "weight", WeightAverageParametrization(input_models, alpha, name, "weight"))
            parametrize.register_parametrization(module, "bias", WeightAverageParametrization(input_models, alpha, name, "bias"))


def model_remove_parametrizations(model):
    for name, module in model.named_modules():
        if(isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d)):
            print("name.module: ", f"{name}.{module}")
            parametrize.remove_parametrizations(module, "weight", leave_parametrized=True)
            parametrize.remove_parametrizations(module, "bias", leave_parametrized=True)


def weighted_average_weights(w, weighted_average_weights):
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        w_avg[key] = torch.zeros(w_avg[key].shape, device=w_avg[key].device)
    for key in w_avg.keys():
        for i in range(len(w)):
            w_avg[key] += w[i][key] * weighted_average_weights[i]
        w_avg[key] = torch.div(w_avg[key], torch.sum(weighted_average_weights))
    return w_avg


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return
