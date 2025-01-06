import torch
import os
import random
import numpy as np

class epochAverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        # calculate avg
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Meter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        # calculate avg
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        # calculate avg
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_model(path, model, optimizer, name="model"):
    if not os.path.exists(path):
        os.mkdir(path)
    torch.save(model, f"{path}/{name}.pth")
    torch.save(model.state_dict(), f'{path}/{name}_params.pth')
    torch.save(optimizer.state_dict(), f'{path}/{name}_optimizer.pth')

def seed_everything(seed):
    """
    Seed all necessary random number generators.
    """
    if seed is None:
        seed = random.randint(1, 10000)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True


def transfer_state_dict(pretrained_dict, model_dict):
    '''
    根据model_dict,去除pretrained_dict一些不需要的参数,以便迁移到新的网络
    url: https://blog.csdn.net/qq_34914551/article/details/87871134
    :param pretrained_dict:
    :param model_dict:
    :return:
    '''
    # state_dict2 = {k: v for k, v in save_model.items() if k in model_dict.keys()}
    state_dict = {}
    for k, v in pretrained_dict.items():
        if k in model_dict.keys():
            if model_dict[k].shape == v.shape:
                state_dict[k] = v
            else:
                shape0, shape1 = model_dict[k].shape
                state_dict[k] = v[:shape0, :shape1]
    return state_dict


def transfer_model(pretrained_model, model):
    '''
    只导入pretrained_model部分模型参数
    tensor([-0.7119,  0.0688, -1.7247, -1.7182, -1.2161, -0.7323, -2.1065, -0.5433,-1.5893, -0.5562]
    update:
        D.update([E, ]**F) -> None.  Update D from dict/iterable E and F.
        If E is present and has a .keys() method, then does:  for k in E: D[k] = E[k]
        If E is present and lacks a .keys() method, then does:  for k, v in E: D[k] = v
        In either case, this is followed by: for k in F:  D[k] = F[k]
    :param pretrained_model:
    :param model:
    :return:
    '''
    pretrained_dict = pretrained_model.state_dict()  # get pretrained dict
    model_dict = model.state_dict()  # get model dict
    # 在合并前(update),需要去除pretrained_dict一些不需要的参数
    pretrained_dict = transfer_state_dict(pretrained_dict, model_dict)
    model_dict.update(pretrained_dict)  # 更新(合并)模型的参数
    model.load_state_dict(model_dict)
    return model