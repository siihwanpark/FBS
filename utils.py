import os
import math
import random
import numpy as np
import argparse
import torch


def set_seed(seed):
    # for reproducibility.
    # note that pytorch is not completely reproducible
    # https://pytorch.org/docs/stable/notes/randomness.html
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.initial_seed()  # dataloader multi processing
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return None


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def global_avgpool2d(x):
    raise NotImplementedError()


def winner_take_all(x, sparsity_ratio):
    raise NotImplementedError()
