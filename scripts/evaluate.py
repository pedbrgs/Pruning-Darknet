import os
import argparse
import subprocess
import numpy as np
import pandas as pd

def evaluate_model(when = 'after'):

    # Opens the temporary file
    f = open('../eval.txt', 'a+')

    # Running evaluation algorithm and saving to temporary file
    if when.lower() == 'after':
        subprocess.call('./darknet detector map dfire.data dfire.cfg weights/dfire_best.weights', shell = True, stdout = f)
    if when.lower() == 'before':
        subprocess.call('./darknet detector map dfire.data dfire.cfg dfire.weights', shell = True, stdout = f)

    # Closing file
    f.close()

def change_hyperparams(imgsize):

    # Opens the file in read-only mode
    f = open('dfire.cfg', 'r')

    # Read lines until EOF
    lines = f.readlines()

    # Loop over the lines
    for i, line in enumerate(lines):
        if 'width' in line:
            lines[i] = 'width=' + str(imgsize) + '\n'
        elif 'height' in line:
            lines[i] = 'height=' + str(imgsize) + '\n'
        elif '^batch =' in line:
            lines[i] = 'batch = 1\n'
        elif '^subdivisions' in line:
            lines[i] = 'subdivisions = 1\n'

    # Opens the file in write-only mode
    f = open('dfire.cfg', 'w')

    # Changing image size in the config file
    f.writelines(lines)
    # Closing file
    f.close()

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

def export_to_csv(method, imgsize, when = 'after'):

    # Opens the file in read-only mode
    f = open('eval.txt', 'r')

    # Read lines until EOF
    lines = f.readlines()

    # Evaluate models after pruning
    if when.lower() == 'after':

        idx = 0
        mAP_train, mAP_valid, F1, ap_smoke, ap_fire, avg_IoU = list(), list(), list(), list(), list(), list()

        # Loop over the lines
        for i, line in enumerate(lines):
            if 'mAP@0.50' in line:
                if idx % 2 != 0:
                    # mAP@0.50 in the validation set
                    mAP_valid.append(float(line.split(' ')[-5].split(',')[0]))
                else:
                    # mAP@0.50 in the training set
                    mAP_train.append(float(line.split(' ')[-5].split(',')[0]))
                idx += 1
            elif 'F1-score' in line and idx%2 != 0:
                # F1-score in the validation set
                F1.append(line.split('=')[-1].split(' ')[-2])
            elif 'ap' in line and idx%2 != 0:
                if 'SMOKE' in line:
                    # AP@0.50 of smoke in the validation set
                    ap_smoke.append(float(line.split('=')[-3].split('%')[-2]))
                if 'FIRE' in line:
                    # AP@0.50 of fire in the validation set
                    ap_fire.append(float(line.split('=')[-3].split('%')[-2]))
            elif 'average IoU' in line and idx%2 != 0:
                # avg IoU in the validation set
                avg_IoU.append(float(line.split('=')[-1].split('%')[0]))
            else:
                pass

        # Closing file
        f.close()

        # Root directory
        os.chdir('/home/adriano/Research/Pruning/')
        # Opening log in read-only mode
        log_filename = ('-').join(method.split('/')) + '_' + str(imgsize) + '.out'
        f = open(log_filename, 'r')

        # Read lines until EOF
        lines = f.readlines()

        BFLOPS = list()

        # Loop over the lines
        for i, line in enumerate(lines):
            if 'Total BFLOPS' in line:
                BFLOPS.append(float(line.split(' ')[-2]))
        BFLOPS = np.unique(BFLOPS)

        evaluation = pd.DataFrame(np.column_stack((mAP_train, mAP_valid, np.array(ap_smoke)/100, np.array(ap_fire)/100, F1, np.array(avg_IoU)/100, sorted(BFLOPS, reverse = True))),
                columns = ['mAP train', 'mAP valid', 'mAP smoke', 'mAP fire', 'F1-Score', 'avg IoU', 'BFLOPS'])

    # Evaluate models before pruning
    if when.lower() == 'before':

        pre_mAP = list()

        # Loop over the lines
        for i, line in enumerate(lines):
            if 'mAP@0.50' in line:
                pre_mAP.append(float(line.split(' ')[-5].split(',')[0]))
        
        # Closing file
        f.close()

        # Root directory
        os.chdir('/home/adriano/Research/Pruning/')

        evaluation = pd.DataFrame(pre_mAP, columns = ['pre-mAP'])

    # Saving to csv file
    csv_filename = ('-').join(method.split('/')) + '_' + str(imgsize) + '.csv'
    evaluation.to_csv(csv_filename)

    # Closing file
    f.close()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--method', type = str, help = 'Pruning method')
    parser.add_argument('--imgsize', type = int, default = 416, help = 'Image size')
    parser.add_argument('--when', type = str, default = 'after', help = 'Evaluate models after pruning or before pruning')
    opt = parser.parse_args()

    # Open root folder
    root = 'Fine-Tuning/' + opt.method + os.sep
    os.chdir(root)

    # Pruned models with pruning rate from 5% to 95%
    folders = np.arange(start = 5, stop = 100, step = 5)

    try:
        # Remove current eval.txt file
        os.remove('eval.txt')
    except:
        pass

    for folder in folders:

        # Open current folder
        subdir = str(folder) + os.sep
        os.chdir(subdir)

        # Change image size, batch size and subdivisions
        change_hyperparams(opt.imgsize)

        if opt.when.lower() == 'after':

            # Evaluates model in training set
            change_set('train')
            evaluate_model(when = opt.when)

            # Evaluates model in validation set
            change_set('valid')
            evaluate_model(when = opt.when)

        if opt.when.lower() == 'before':

            # Evaluates model just in validation set
            change_set('valid')
            evaluate_model(when = opt.when)

        # Returns to root folder
        os.chdir('../')

    export_to_csv(opt.method, opt.imgsize, opt.when)
