import os
import argparse
import subprocess
import numpy as np
import pandas as pd

def training_model(pruning_rate, layer):

    # Opens the temporary file
    f = open('../eval.txt', 'a+')

    # Running training algorithm and saving results to temporary file
    weights = 'dfire' + str(pruning_rate) + '.conv.' + str(layer)
    command = './darknet detector train dfire.data dfire.cfg ' + weights + ' -dont_show -map'
    subprocess.call(command, shell = True, stdout = f)

    # Closing file
    f.close()

    print('\n[DONE] Fine-tuning of the model pruned by ' + str(pruning_rate) + '%\n')

def pre_weights(pruning_rate, layer):

    # Opens the temporary file
    f = open('../eval.txt', 'a+')

    # Running freezing algorithm and saving results to temporary file
    weights = 'dfire' + str(pruning_rate) + '.conv.' + str(layer)
    command = './darknet partial dfire.cfg dfire.weights ' + weights + ' ' + str(layer)
    subprocess.call(command, shell = True, stdout = f)

    # Closing file
    f.close()

    print('\n[DONE] Pre-weights of the model pruned by ' + str(pruning_rate) + '%\n')


def change_set(set):

    # Opens the file in read-only mode
    f = open('dfire.data', 'r')

    # Reads lines until EOF
    lines = f.readlines()

    # Loop over the lines
    for i, line in enumerate(lines):
        if 'valid' in line:
          lines[i] = 'valid = data/dfire_' + set + '.txt\n'

    # Opens the file in write-only mode
    f = open('dfire.data', 'w')

    # Changing validation set in the data file
    f.writelines(lines)
    # Closing file
    f.close()

def change_hyperparams(imgsize, iter, lr, steps):

    # Opens the file in read-only mode
    f = open('dfire.cfg', 'r')

    # Read lines until EOF
    lines = f.readlines()

    # Loop over the lines
    for i, line in enumerate(lines):
        if 'width' in line:
            lines[i] = 'width = ' + str(imgsize) + '\n'
        elif 'height' in line:
            lines[i] = 'height = ' + str(imgsize) + '\n'
        elif 'steps = ' in line:
            lines[i] = 'steps = ' + steps + '\n'
        elif 'learning_rate' in line:
            lines[i] = 'learning_rate = ' + str(lr) + '\n'
        elif 'max_batches' in line:
            lines[i] = 'max_batches = ' + str(iter) + '\n'
        elif '^batch =' in line:
            lines[i] = 'batch = 64\n'
        elif '^subdivisions' in line:
            lines[i] = 'subdivisions = 16\n'

    # Opens the file in write-only mode
    f = open('dfire.cfg', 'w')

    # Changing image size in the config file
    f.writelines(lines)
    # Closing file
    f.close()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--method', type = str, help = 'Pruning method')
    parser.add_argument('--imgsize', type = int, default = 416, help = 'Image size')
    parser.add_argument('--lr', type = float, default = 0.001, help = 'Learning rate')
    parser.add_argument('--tuning-iter', type = int, default = 8000, help = 'Number of fine-tuning iterations')
    parser.add_argument('--layer', type = int, default = 106, help = 'Weights frozen up to this layer for fine-tuning')
    parser.add_argument('--steps', type = str, default = '6400,7200', help = 'At these iterations the learning rate will be multiplied by scales factor (0.1 by default)')
    opt = parser.parse_args()

    # Open root folder
    root = 'Fine-Tuning/' + opt.method + os.sep
    os.chdir(root)

    # Pruned models with pruning rate from 5% to 95%
    folders = np.arange(start = 5, stop = 100, step = 5)
    # Partial weights
    weights = np.arange(start = 1000, stop = 9000, step = 1000)

    try:
        # Remove current eval.txt file
        os.remove('eval.txt')
    except:
        pass

    for folder in folders:

        # Open current folder
        subdir = str(folder) + os.sep
        os.chdir(subdir)

        # Change hyperparameters
        change_hyperparams(opt.imgsize, opt.tuning_iter, opt.lr, opt.steps)
        # Change validation set
        change_set('valid')
        # Freezing layers to generate pre-weights
        pre_weights(folder, opt.layer)
        # Training pruned model
        training_model(folder, opt.layer)

        # Remove partial weights
        os.chdir('weights/')
        for w in weights:
            os.remove('dfire_' + str(w) + '.weights')
        os.remove('dfire_last.weights')
        os.remove('dfire_final.weights')

        # Returns to root folder
        os.chdir('../../')