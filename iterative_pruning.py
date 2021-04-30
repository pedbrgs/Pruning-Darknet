import os
import shutil
import argparse
import numpy as np
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
    parser.add_argument('--lr', type = float, default = 0.005, help = 'learning rate for fine-tuning')
    parser.add_argument('--tuning-iter', type = int, default = 30000, help = 'fine-tuning iterations')
    parser.add_argument('--technique', type = str, help = 'pruning technique')
    parser.add_argument('--pruning-iter', type = int, default = 5, help = 'pruning iterations')
    parser.add_argument('--pruning-rate', type = float, help = 'pruning rate per iteration')
    parser.add_argument('--network', type = str, default = 'YOLOv4', help = 'network version')
    parser.add_argument('--measure', type = str, help = 'correlation measure for clustering approach')
    parser.add_argument('--pool-type', type = str, default = 'max', help = 'pooling operation')
    parser.add_argument('--n-components', type = int, help = 'number of components of projection method')
    parser.add_argument('--num-classes', type = int, default = 3, help = 'number of classes of the output variable for projection')
    parser.add_argument('--perc-samples', type = float, default = 1.0, help = 'percentage of samples to train the projection model')

    opt = parser.parse_args()
    print(opt)

    # Create temporary folder
    os.makedirs('./temp/', exist_ok = True)
    # Move .cfg file to temporary folder
    shutil.copy2(src = opt.cfg, dst = './temp/' + opt.cfg)
    opt.cfg = './temp/' + opt.cfg
    # Move .weights file to temporary folder
    shutil.copy2(src = opt.weights, dst = './temp/' + opt.weights)
    opt.weights = './temp/' + opt.weights

    print('Pruning technique:', opt.technique)

    # Iterative pruning
    for pruning_iter in range(opt.pruning_iter):

        print('Pruning iteration:', pruning_iter+1)

        # Loading Darknet model in PyTorch
        enable_cfg(opt.cfg, framework = 'PyTorch')
        model = YOLO(opt.cfg, opt.data, opt.names, opt.weights, opt.img_size)

        print('Prunable layers:', len(to_prune(model)))
        print('Prunable filters:', prunable_filters(model))

        # To prune using a projection-based method, you must generate the X and Y variables first
        if opt.technique.upper() in ['PLS-VIP-SINGLE', 'PLS-VIP-MULTI', 'CCA-CV-MULTI', 'PLS-LC-MULTI']:
            X, Y = get_variables(model, opt.data, opt.img_size, opt.num_classes, opt.pool_type, opt.perc_samples)

        # Calculate correlation matrices
        if opt.technique.upper() == 'HAC':
            CCM = correlation_matrices(model, opt.measure)
            print('Number of correlation matrices:', len(CCM))
            print('Dimension of first correlation matrix:', CCM[0].shape)
            print('Dimension of last correlation matrix:', CCM[-1].shape)

        # Prune network with criteria-based method
        if opt.technique.upper() in ['L0', 'L1', 'L2', 'L-INF']:
            model = criteria_based_pruning(model, opt.pruning_rate, opt.technique)
        # Prune network with projection-based method
        elif opt.technique.upper() in ['PLS-VIP-SINGLE', 'PLS-VIP-MULTI', 'CCA-CV-MULTI', 'PLS-LC-MULTI']:
            model = projection_based_pruning(model, opt.pruning_rate, opt.technique, X, Y, opt.n_components)
        # Prune network with wrapper-based method
        elif opt.technique.upper() == 'HAC':
            model = wrapper_based_pruning(model, opt.pruning_rate, opt.technique, CCM)
        # Prune network randomly
        elif opt.technique.upper() in ['RANDOM', 'FROM-SCRATCH']:
            model = random_pruning(model, opt.pruning_rate, -1)
        else:
            raise AssertionError('The technique %s does not exist.' % (opt.technique))

        # Save network configuration file and network weights
        save_weights(model, path = opt.weights)
        version = int(opt.network.split('v')[-1])    
        model_to_cfg(model, cfg = opt.cfg, mode = 'train', version = version)

        # Deleting current model
        del model

        # Fine-tuning
        fine_tuning(filename = opt.names.split('.')[0],
                    technique = opt.technique,
                    pruning_rate = opt.pruning_rate,
                    img_size = opt.img_size,
                    lr = opt.lr,
                    tuning_iter = opt.tuning_iter,
                    layer = 161,
                    steps = str(int(opt.tuning_iter*0.8)) + ',' +
                            str(int(opt.tuning_iter*0.9)))