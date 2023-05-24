import numpy as np
import os
import torch
import torch.backends.cudnn as cudnn
import random

def set_seed(args):
    # fix randomseed for reproducing the results
    print('Setting random seed for reproductivity..')
    random_seed = args.random_seed
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    cudnn.benchmark = not args.not_cuda_benchmark