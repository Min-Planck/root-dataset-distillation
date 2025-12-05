import random
import os
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from models import convnet as CN
from models import resnet as RN

def define_model(args, num_classes, mode='pool', e_model=None):
    '''Obtain model for training and validating
    '''

    if e_model:
        model = e_model
    else:
        if mode == 'pool': 
            model_pool = ['convnet', 'resnet10', 'resnet18']
            model = random.choice(model_pool)
            print('Random model: {}'.format(model))
        elif mode == 'single':
             model = args.eval_model[0]

    if args.data == 'mnist' or args.data == 'fmnist':
        nch = 1
    else:
        nch = 3

    if model == 'convnet':
        return CN.ConvNet(num_classes, channel=nch)
    elif model == 'resnet10':
        return RN.ResNet(args.data, 10, num_classes, nch=nch)
    elif model == 'resnet18':
        return RN.ResNet(args.data, 18, num_classes, nch=nch)
    elif model == 'resnet34':
        return RN.ResNet(args.data, 34, num_classes, nch=nch)
    elif model == 'resnet50':
        return RN.ResNet(args.data, 50, num_classes, nch=nch)
    elif model == 'resnet101':
        return RN.ResNet(args.data, 101, num_classes, nch=nch)
    
def load_data(args):
    '''Obtain data
    '''
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    if args.data == 'cifar10':
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.491, 0.482, 0.447), (0.202, 0.199, 0.201))
        ])
        trainset = datasets.CIFAR10(root=args.data_dir, train=True, download=True,
                                    transform=transform_train)
        testset = datasets.CIFAR10(root=args.data_dir, train=False, download=True,
                                   transform=transform_test)
    elif args.data == 'svhn':
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.437, 0.444, 0.473), (0.198, 0.201, 0.197))
        ])
        trainset = datasets.SVHN(os.path.join(args.data_dir, 'svhn'),
                                 split='train',
                                 download=True,
                                 transform=transform_train)
        testset = datasets.SVHN(os.path.join(args.data_dir, 'svhn'),
                                split='test',
                                download=True,
                                transform=transform_test)
    elif args.data == 'fmnist':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.286,), (0.353,))
        ])

        trainset = datasets.FashionMNIST(args.data_dir, train=True, download=True,
                                 transform=transform_train)
        testset = datasets.FashionMNIST(args.data_dir, train=False, download=True,
                                 transform=transform_train)

    

    return trainset, testset
