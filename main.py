import os
import sys
import time
import random
import argparse
import numpy as np
import torch

from generative_distillation.codim import start_codim
from generative_distillation.dim import start_dim
from pixel_optimization.dsa import start_dsa
from pixel_optimization.dm import start_dm
from pixel_optimization.cafe import start_cafe
from pixel_optimization.dc import start_dc

from common import load_data 

def str2bool(v):
    """Cast string to boolean
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()

    parser.add_argument('--algorithm', type=str, default='codim')
    parser.add_argument('--ipc', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--epochs-eval', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr-img', type=float, default=0.1, help="Learning rate for updating synthetic images")
    parser.add_argument('--eval-lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--eval-model', type=str, nargs='+', default=['convnet'])
    parser.add_argument('--dim-noise', type=int, default=100)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--print-freq', type=int, default=50)
    parser.add_argument('--eval-interval', type=int, default=10)
    parser.add_argument('--test-interval', type=int, default=200)
    parser.add_argument('--lambda-div', type=float, default=0.1, help='Weight for Generator Diversity Loss')
    parser.add_argument('--lambda-ctr', type=float, default=0.1, help='Weight for Discriminator Contrastive Learning')

    parser.add_argument('--data', type=str, default='cifar10')
    parser.add_argument('--num-classes', type=int, default=10)
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--output-dir', type=str, default='./results/')
    parser.add_argument('--logs-dir', type=str, default='./logs/')
    parser.add_argument('--aug-type', type=str, default='color_crop_cutout')
    parser.add_argument('--mixup-net', type=str, default='cut')
    parser.add_argument('--bias', type=str2bool, default=False)
    parser.add_argument('--fc', type=str2bool, default=False)
    parser.add_argument('--mix-p', type=float, default=-1.0)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--tag', type=str, default='test')
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--lambda-1', type=float, default=0.04, help='cafe')
    parser.add_argument('--lambda-2', type=float, default=0.03, help='cafe')
    parser.add_argument('--first-weight', type=float, default=1.0, help='cafe')
    parser.add_argument('--second-weight', type=float, default=1.0, help='cafe')
    parser.add_argument('--third-weight', type=float, default=0.1, help='cafe')
    parser.add_argument('--fourth-weight', type=float, default=0.1, help='cafe') 
    parser.add_argument('--inner-weight', type=float, default=0.01, help='cafe')

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    args.output_dir = args.output_dir + args.tag + '/' + args.algorithm + '/' + args.data

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if not os.path.exists(args.output_dir + '/outputs'):
        os.makedirs(args.output_dir + '/outputs')
    
    print(args)

    trainset, testset = load_data(args)
    algorithm = args.algorithm.lower()

    if algorithm == 'dc':
        func = start_dc
    elif algorithm == 'dsa':
        func = start_dsa
    elif algorithm == 'dm':
        func = start_dm
    elif algorithm == 'cafe':
        func = start_cafe
    elif algorithm == 'dim':
        func = start_dim
    elif algorithm == 'codim':
        func = start_codim

    func(args, trainset, testset)
    

    