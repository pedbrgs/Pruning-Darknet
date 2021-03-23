import torch
from models import *
from utils.utils import *
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

def YOLO(cfg, data, names, weights, img_size = 416, device = False):

    if device:
        device = torch_utils.select_device()
        model = Darknet(cfg, img_size).to(device)
    else:
        model = Darknet(cfg, img_size)
    
    # PyTorch format
    if weights.endswith('.pt'):
        print('PyTorch weights loaded')
        model.load_state_dict(torch.load(weights, map_location = device)['model'])
    # Darknet format
    elif weights.endswith('.weights'):
        print('Darknet weights loaded')
        load_darknet_weights(model, weights)
    else:
        print('Random weights loaded')    
    
    return model

def Optimizer(model, hyp, opt, verbose = False):

    # Optimizer parameter groups
    pg0, pg1, pg2 = [], [], []  
    for k, v in dict(model.named_parameters()).items():
        # biases
        if '.bias' in k:
            pg2 += [v]  
        # apply weight_decay
        elif 'Conv2d.weight' in k:
            pg1 += [v]  
        # all else
        else:
            pg0 += [v]

    # Optimizer
    if opt['optimizer'].lower() == 'adam':
        optimizer = optim.Adam(pg0, lr = hyp['lr0'])
    elif opt['optimizer'].lower() == 'adamax':
        optimizer = optim.Adamax(pg0, lr = hyp['lr0'])
    else:
        optimizer = optim.SGD(pg0, lr = hyp['lr0'], momentum = hyp['momentum'], nesterov = hyp['nesterov'])
        
    # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  
    # add pg2 (biases)
    optimizer.add_param_group({'params': pg2})  
    del pg0, pg1, pg2

    if verbose:
        print(optimizer)

    return optimizer

def Scheduler(optimizer, hyp, opt, verbose = False):

    # Scheduler https://github.com/ultralytics/yolov3/issues/238

    if opt['scheduler'].lower() == 'cos':
        # Cosine https://arxiv.org/pdf/1812.01187.pdf
        lf = lambda x: (((1 + math.cos(x * math.pi / opt['epochs'])) / 2) ** 1.0) * 0.95 + 0.05
        if verbose:
            print('Policy: Cosine')
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda = lf)
    elif opt['scheduler'].lower() == 'invexp':
        # Inverse exp ramp
        if verbose:
            print('Policy: Inverse Exponential')
        lf = lambda x: 1 - 10 ** (hyp['lrf'] * (1 - x / opt['epochs']))
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda = lf) 
    else:
        # MultiStep (Darknet default)
        if verbose:
            print('Policy: MultiStep')
            print('Steps:', round(0.8*opt['epochs']), round(0.9*opt['epochs']))
        scheduler = lr_scheduler.MultiStepLR(optimizer, [round(opt['epochs'] * x) for x in [0.8, 0.9]], 0.1)

    scheduler.last_epoch = opt['start_epoch'] - 1

    return scheduler