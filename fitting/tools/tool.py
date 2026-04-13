import os
import pathlib
import datetime
import numpy as np
import random
import torch
import inspect
import math

def rescale(old_value, new_lb, new_ub, old_lb=-1., old_ub=1.):
    new_value = (new_ub - new_lb) * (old_value - old_lb) / (old_ub - old_lb) + new_lb
    return new_value

def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly
    Args:
        optimizer: (torch.optim) optimizer
        epoch: (int) current epoch
        total_num_epochs: (int) total number of epochs
        initial_lr: (float) initial learning rate
    """
    learning_rate = initial_lr - (initial_lr * ((epoch - 1) / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate

def get_grad_norm(parameters):
    """Get gradient norm."""
    sum_grad = 0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        sum_grad += parameter.grad.norm() ** 2
    return math.sqrt(sum_grad)

def check(value):
    """Check if value is a numpy array, if so, convert it to a torch tensor."""
    output = torch.from_numpy(value) if isinstance(value, np.ndarray) else value
    return output

def compute_reward_to_go(rews):
    n = len(rews)
    rtgs = np.zeros_like(rews)
    for i in reversed(range(n)):
        rtgs[i] = rews[i] + (rtgs[i+1] if i+1 < n else 0)
    return rtgs

def json_default(o):
    if inspect.isfunction(o) or inspect.isclass(o):
        return f'<from {o.__module__} import {o.__name__}>'
    else:
        return '<not serializable>'

# set project root directory as current working directory
def set_project_root_as_working_directory(file):
    level = 0
    while True:
        dir = pathlib.Path(file).parents[level]
        if 'fitting' == dir.name:
            os.chdir(dir)
            return
        level += 1


def current_timestamp():
    now = datetime.datetime.now()
    timestamp = str(now.year) + '-' + str(now.month).zfill(2) + str(now.day).zfill(2) + '/' + \
        str(now.hour).zfill(2) + str(now.minute).zfill(2) + \
        '-' + str(now.second).zfill(2)
    print(f'current timestamp is {timestamp}')
    return timestamp


def get_seeds(num_seeds):
    rng = np.random.default_rng()
    seeds = rng.choice(100000, size=num_seeds, replace=False)
    print(f'seeds are {seeds}')
    seeds = seeds.tolist()
    return seeds


def set_seed(seed):
    # print(f'pid {os.getpid()}')
    # print(f'seed {seed}')
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# def set_seed(seed=None):
#     if seed is None:
#         rng = np.random.default_rng()
#         seed = int(rng.integers(1000, 10000))
#     print(f'pid {os.getpid()}')
#     print(f'seed {seed}')
#     random.seed(seed)
#     np.random.seed(seed)
#     os.environ["PYTHONHASHSEED"] = str(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)


def set_sub_seed(main_seed, sub_seed):
    # different threads have dirrerent seeds        
    seed = None if main_seed is None else main_seed+5+7*sub_seed  
    set_seed(seed)    
    
    
def t2n(value):
    """Convert torch.Tensor to numpy.ndarray."""
    return value.clone().detach().cpu().numpy()


def init_device(args):
    """Init device.
    Args:
        args: (dict) arguments
    Returns:
        device: (torch.device) device
    """
    if 'cuda' in args['train_device'] and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device(args['train_device'])
        # device = torch.device('cuda')
        if args["cuda_deterministic"]:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
    torch.set_num_threads(args.get('torch_threads', 1))
    return device
