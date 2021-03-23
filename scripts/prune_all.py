import os
import argparse
from models import *
from utils.utils import *
from utils.build import *
from pruning.utils import *
from pruning.techniques import *

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg', type = str, default = 'yolov4.cfg', help = '*.cfg path')
    parser.add_argument('--data', type = str, default = 'yolov4.data', help = '*.data path')
    parser.add_argument('--names', type = str, default = 'yolov4.names', help = '*.names path')
    parser.add_argument('--weights', type = str, default = 'yolov4.weights', help = '*.weights path')
    parser.add_argument('--img-size', type = int, default = 416, help = 'input image size')
    parser.add_argument('--technique', type = str, help = 'pruning technique')
    parser.add_argument('--network', type = str, default = 'YOLOv4', help = 'network version')
    parser.add_argument('--variables', type = str, default = 'variables.npy', help = 'variables.npy path')
    parser.add_argument('--n-components', type = int, help = 'number of components of projection method')

    opt = parser.parse_args()
    print(opt)

    # Create output folder
    os.mkdir(opt.technique + os.sep)

    # Pruning rates from 5% to 95% with step by 5%
    pruning_rates = np.arange(start = 0.05, stop = 1, step = 0.05)

    for pruning_rate in pruning_rates:

        # Loading Darknet model in PyTorch
        enable_cfg(opt.cfg, framework = 'PyTorch')
        model = YOLO(opt.cfg, opt.data, opt.names, opt.weights, opt.img_size)

        # Prune network with criteria-based method
        if opt.technique.upper() in ['L0', 'L1', 'L2', 'L-INF']:
            model = criteria_based_pruning(model, pruning_rate, opt.technique)
        # Prune network with projection-based method
        elif opt.technique.upper() in ['PLS-VIP-SINGLE', 'PLS-VIP-MULTI', 'CCA-CV-MULTI', 'PLS-LC-MULTI']:
            try:
                with open(opt.variables, 'rb') as f:
                    X = np.load(f)
                    Y = np.load(f)
            except:
                raise AssertionError('To prune using projection-based methods, you must generate the X and Y variables first.')
            model = projection_based_pruning(model, pruning_rate, opt.technique, X, Y, opt.n_components)
        # Prune network randomly
        elif opt.technique.upper() == 'RANDOM':
            model = random_pruning(model, pruning_rate, -1)
        else:
            raise AssertionError('The technique %s does not exist.' % (opt.technique))

        # Save network configuration file and network weights
        save_weights(model, path = opt.technique + os.sep + 'pruned' + str(pruning_rate*100) + opt.weights)
        version = int(opt.network.split('v')[-1])    
        model_to_cfg(model, cfg = opt.technique + os.sep + 'pruned' + str(pruning_rate*100) + opt.cfg, mode = 'train', version = version)
    
        del model