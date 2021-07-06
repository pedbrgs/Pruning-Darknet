import os
import shutil
import argparse
import subprocess
import numpy as np

def training_model(filename, technique, pruning_rate, layer):

    """ Training the pruned model """

    # Opens the temporary file
    f = open('../eval.txt', 'a+')

    # Training with pre-trained weights
    if technique.upper() != 'FROM-SCRATCH':
        weights = filename + str(pruning_rate) + '.conv.' + str(layer)
        command = './darknet detector train ' + filename + '.data ' + filename + '.cfg ' + weights + ' -dont_show -map'
    # Training from scratch
    else:
        command = './darknet detector train ' + filename + '.data ' + filename + '.cfg -dont_show -map'

    # Running training algorithm and saving results to temporary file
    subprocess.call(command, shell = True, stdout = f)

    # Closing file
    f.close()

    print('\n[DONE] Fine-tuning of the model pruned by ' + str(pruning_rate) + '%\n')

def pre_weights(filename, pruning_rate, layer):

    """ Generates a pre-weight from a trained weight """

    # Opens the temporary file
    f = open('../eval.txt', 'a+')

    # Running freezing algorithm and saving results to temporary file
    weights = filename + str(pruning_rate) + '.conv.' + str(layer)
    command = './darknet partial ' + filename + '.cfg ' + filename + '.weights ' + weights + ' ' + str(layer)
    subprocess.call(command, shell = True, stdout = f)

    # Closing file
    f.close()

    print('\n[DONE] Pre-weights of the model pruned by ' + str(pruning_rate) + '%\n')


def valid_set(filename, set):

    """ Changes validation set in the .data file """

    # Opens the file in read-only mode
    f = open(filename + '.data', 'r')

    # Reads lines until EOF
    lines = f.readlines()

    # Loop over the lines
    for i, line in enumerate(lines):
        if 'valid' in line:
          lines[i] = 'valid = data/' + filename + '_' + set + '.txt\n'

    # Opens the file in write-only mode
    f = open(filename + '.data', 'w')

    # Changing validation set in the data file
    f.writelines(lines)
    # Closing file
    f.close()

def hyperparams(filename, img_size, iter, lr, steps):

    """ Changes hyperparameters of the .cfg file """

    # Opens the file in read-only mode
    f = open(filename + '.cfg', 'r')

    # Read lines until EOF
    lines = f.readlines()

    # Loop over the lines
    for i, line in enumerate(lines):
        if 'width' in line:
            lines[i] = 'width = ' + str(img_size) + '\n'
        elif 'height' in line:
            lines[i] = 'height = ' + str(img_size) + '\n'
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
    f = open(filename + '.cfg', 'w')

    # Changing image size in the config file
    f.writelines(lines)
    # Closing file
    f.close()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--method', type = str, help = 'Pruning method')
    parser.add_argument('--imgsize', type = int, default = 416, help = 'Image size')
    parser.add_argument('--dataset', type = str, default = 'dfire', help = 'Dataset name')
    parser.add_argument('--pruning-rate', type = int, default = 80, help = 'Pruning rate (in percentage)')
    parser.add_argument('--lr', type = float, default = 0.01, help = 'Learning rate')
    parser.add_argument('--layer', type = int, default = 161, help = 'Weights frozen up to this layer for fine-tuning')
    opt = parser.parse_args()

    # Open root folder
    root = 'Fine-Tuning/' + opt.method + os.sep + str(opt.pruning_rate) + os.sep
    os.chdir(root)

    # Create temporary folder
    os.makedirs('./convergence/', exist_ok = True)

    # Fine tuning iterations
    iterations = np.arange(start = 5000, stop = 30000, step = 5000)

    try:
        # Remove current eval.txt file
        os.remove('eval.txt')
    except:
        pass

    for iter in iterations:

        # Steps
        steps = str(int(iter*0.8)) + ',' + str(int(iter*0.9))

        # Partial weights
        weights = np.arange(start = 1000, stop = iter + 1000, step = 1000)

        # Change hyperparameters
        hyperparams(opt.dataset, opt.imgsize, iter, opt.lr, steps)
        # Change validation set
        valid_set(opt.dataset, 'valid')
        # Pre-trained weights
        if opt.method.upper() != 'FROM-SCRATCH':
            # Freezing layers to generate pre-weights
            pre_weights(opt.dataset, opt.pruning_rate, opt.layer)

        # Training pruned model
        training_model(opt.dataset, opt.method, opt.pruning_rate, opt.layer)

        # Remove partial weights
        os.chdir('weights/')
        for w in weights:
            try:
                os.remove(opt.dataset + '_' + str(w) + '.weights')
            except:
                pass
        os.remove(opt.dataset + '_last.weights')
        os.remove(opt.dataset + '_final.weights')
        shutil.copy2(src = opt.dataset + '_best.weights', dst = '../convergence/' + opt.dataset + '_best_' + str(int(iter)) + '.weights')

        # Returns to root folder
        os.chdir('../')