import torch
import random
import collections
import pandas as pd
from tqdm import tqdm
from utils.utils import *
from pruning.utils import *
from utils.parse_config import *
from torch.utils.data import DataLoader
from utils.datasets import *
from sklearn.cross_decomposition import CCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.cluster import AgglomerativeClustering
from scipy.stats import pearsonr, spearmanr, kendalltau

def to_prune(model):

    """ Returns the indexes of the convolutional blocks that can be pruned."""

    blocks = list()

    for i in range(len(model.module_list)):
        try:
            for j in range(len(model.module_list[i])):
                block = str(model.module_list[i][j]).split('(')[0]
                next_block = str(model.module_list[i+1]).split('(')[0]
                previous_block = str()
                # It must be a sequential block containing "Conv2d + BatchNorm2d + LeakyReLU" and that does not precede a YOLO layer
                if block == 'Conv2d' and i+1 not in model.yolo_layers:
                    blocks.append(i)
        except:
            pass

    return blocks

def find_routes(model):

    """ Find skip connections (routes) in the model. """

    routes = collections.defaultdict(list)

    for i in range(len(model.module_list)):
    
        block = str(model.module_list[i]).split('(')[0]

        if block == 'FeatureConcat':
    
            layers = model.module_list[i].layers
    
            for layer in layers:
  
                # Relative or true index
                idx = i + layer if layer < 0 else layer
    
                if str(model.module_list[idx]).split('(')[0] == 'Sequential':
                    # i+1 because it is the convolutional layer after the FeatureConcat
                    if str(model.module_list[i+1]).split('(')[0] == 'Sequential':
                        routes[idx].append(i+1)
                else:
                    # Closest previous convolutional layer
                    for k in range(idx, 0, -1):
                        if str(model.module_list[k]).split('(')[0] == 'Sequential' and str(model.module_list[i+1]).split('(')[0] == 'Sequential':
                            routes[k].append(i+1)
                            break

    return routes

def get_filter_index(model, block, filter, routes):

    """ Get relative filter index of the convolutional layer that is in FeatureConcat with some other layer(s). """

    filters = list()
    n_filters = list()
    links = collections.defaultdict(list)

    # Key: index of the layer that is the result of the concatenation, value: indexes of the layers that were concatenated
    for key, values in routes.items():
        for value in values:
            links[value].append(key)

    # Get number of filters in each layer that concatenate
    for index, value in enumerate(links[list(set(routes[block]))[0]]):
        n_filters.append(model.module_list[value][0].out_channels)
        if value == block:
            idx = index

    # Get index of the filter that will be removed from the layer resulting from the concatenation
    if len(links[list(set(routes[block]))[0]]) != len(set(links[list(set(routes[block]))[0]])):
        for i in range(len(links[list(set(routes[block]))[0]])):
            filters.append(sum(n_filters[:(i)])+filter)
    else:
        filters.append(sum(n_filters[:(idx)])+filter)

    return filters


def prunable_filters(model):

    """ Computes number of prunable filters. """

    n_filters = 0

    blocks = to_prune(model)

    for block in blocks:
        n_filters += model.module_list[block][0].weight.data.shape[0]

    return n_filters

def get_layer_info(layer):

    """ Extracts information that makes up the layer, as well as its weights and bias. """

    hyperparameters = dict()
    parameters = dict()

    # Convolutional layer
    if str(layer).split('(')[0] == 'Conv2d':
        
        hyperparameters['in_channels'] = layer.in_channels
        hyperparameters['out_channels'] = layer.out_channels
        hyperparameters['kernel_size'] = layer.kernel_size
        hyperparameters['stride'] = layer.stride
        hyperparameters['padding'] = layer.padding

        if layer.bias is not None:
            hyperparameters['bias'] = True
            parameters['bias'] = layer.bias.clone()
        else:
            hyperparameters['bias'] = False
            parameters['bias'] = None
        parameters['weight'] = layer.weight.clone()

    # Batch normalization layer
    elif str(layer).split('(')[0] == 'BatchNorm2d':

        hyperparameters['num_features'] = layer.num_features
        hyperparameters['eps'] = layer.eps
        hyperparameters['momentum'] = layer.momentum
        hyperparameters['affine'] = layer.affine
        hyperparameters['track_running_stats'] = layer.track_running_stats

        parameters['bias'] = layer.bias.clone()
        parameters['weight'] = layer.weight.clone()
        
    return hyperparameters, parameters

def replace_layer(model, block, layer):

    """ Replaces original layer with pruned layer. """

    if str(layer).split('(')[0] == 'Conv2d':
        model.module_list[block][0] = layer

    elif str(layer).split('(')[0] == 'BatchNorm2d':
        model.module_list[block][1] = layer

    return model

def remove_filter(parameters, filter, name = 'weight', channels = 'output'):

    """ Removes convolutional filter from a layer. """

    if channels == 'output':

        if filter != 0:

            head_tensor = parameters[name][:filter]
            tail_tensor = parameters[name][filter+1:]
            parameters[name].data = torch.cat((head_tensor, tail_tensor), axis = 0)

        else:
            parameters[name].data = parameters[name][filter+1:]


    elif channels == 'input':

        if filter != 0:

            head_tensor = parameters[name][:,:filter]
            tail_tensor = parameters[name][:,filter+1:]
            parameters[name].data = torch.cat((head_tensor, tail_tensor), axis = 1)

        else:
            parameters[name].data = parameters[name][:,filter+1:]

    return parameters

def single_pruning(model, block, filter):

    """ Pruning a single convolutional filter of the YOLOv3 model. """

    # Log file
    log = open('pruned_filters.txt', 'a+')

    # Find routes in the model
    routes = find_routes(model)

    # Get information from the current convolutional layer
    hyperparameters, parameters = get_layer_info(model.module_list[block][0])

    # Creates a replica of the convolutional layer to perform pruning
    pruned_conv_layer = torch.nn.Conv2d(in_channels = hyperparameters['in_channels'],
                                        out_channels = hyperparameters['out_channels']-1,
                                        kernel_size = hyperparameters['kernel_size'],
                                        stride = hyperparameters['stride'],
                                        padding = hyperparameters['padding'],
                                        bias = False if parameters['bias'] is None else True                                  
                                        )
    
    # Removes convolutional filter
    parameters = remove_filter(parameters, filter, name = 'weight', channels = 'output')

    # Updates pruned convolutional layer
    pruned_conv_layer.weight.data = parameters['weight'].data
    pruned_conv_layer.weight.requires_grad = True

    if parameters['bias'] is not None:
        parameters = remove_filter(parameters, filter, name = 'bias', channels = 'output')
        pruned_conv_layer.bias.data = parameters['bias'].data
        pruned_conv_layer.bias.requires_grad = True

    # Exchanges the original layer with the pruned layer
    model = replace_layer(model, block, pruned_conv_layer)

    # If the block contains more than one layer, convolutional layer is the first
    if len(model.module_list[block]) > 1:

        # Get information from the current batch normalization layer
        hyperparameters, parameters = get_layer_info(model.module_list[block][1])

        # Creates a replica of the batch normalization layer to perform pruning
        pruned_batchnorm_layer = torch.nn.BatchNorm2d(num_features = hyperparameters['num_features']-1,
                                                      eps = hyperparameters['eps'],
                                                      momentum = hyperparameters['momentum'],
                                                      affine = hyperparameters['affine'],
                                                      track_running_stats = hyperparameters['track_running_stats']
                                                      )
        
        # Removes filter
        parameters = remove_filter(parameters, filter, name = 'weight', channels = 'output')
        parameters = remove_filter(parameters, filter, name = 'bias', channels = 'output') 

        pruned_batchnorm_layer.weight.data = parameters['weight'].data
        pruned_batchnorm_layer.weight.requires_grad = True

        pruned_batchnorm_layer.bias.data = parameters['bias'].data
        pruned_batchnorm_layer.bias.requires_grad = True

        # Exchanges the original layer with the pruned layer
        model = replace_layer(model, block, pruned_batchnorm_layer)

    # If the next block is also sequential
    if str(model.module_list[block+1]).split('(')[0] == 'Sequential':

        # Get information from the next convolutional layer
        hyperparameters, parameters = get_layer_info(model.module_list[block+1][0])

        # Creates a replica of the convolutional layer to perform pruning
        pruned_conv_layer = torch.nn.Conv2d(in_channels = hyperparameters['in_channels']-1,
                                            out_channels = hyperparameters['out_channels'],
                                            kernel_size = hyperparameters['kernel_size'],
                                            stride = hyperparameters['stride'],
                                            padding = hyperparameters['padding'],
                                            bias = False if parameters['bias'] is None else True                                  
                                            )
        
        # Removes convolutional filter
        parameters = remove_filter(parameters, filter, name = 'weight', channels = 'input')

        # Updates pruned convolutional layer
        pruned_conv_layer.weight.data = parameters['weight'].data
        pruned_conv_layer.weight.requires_grad = True

        # Exchanges the original layer with the pruned layer
        model = replace_layer(model, block+1, pruned_conv_layer)

    # If the next block is WeightedFeatureFusion
    elif str(model.module_list[block+1]).split('(')[0] == 'WeightedFeatureFusion':

        # Get information from the convolutional layer that comes immediately after the WeightedFeatureFusion
        hyperparameters, parameters = get_layer_info(model.module_list[block+2][0])

        # Creates a replica of the convolutional layer to perform pruning
        pruned_conv_layer = torch.nn.Conv2d(in_channels = hyperparameters['in_channels']-1,
                                            out_channels = hyperparameters['out_channels'],
                                            kernel_size = hyperparameters['kernel_size'],
                                            stride = hyperparameters['stride'],
                                            padding = hyperparameters['padding'],
                                            bias = False if parameters['bias'] is None else True                                  
                                            )
        
        # Removes convolutional filter
        parameters = remove_filter(parameters, filter, name = 'weight', channels = 'input')

        # Updates pruned convolutional layer
        pruned_conv_layer.weight.data = parameters['weight'].data
        pruned_conv_layer.weight.requires_grad = True

        # Exchanges the original layer with the pruned layer
        model = replace_layer(model, block+2, pruned_conv_layer)
    
    if block in routes.keys():

        # Output layer index of the FeatureConcat
        dest = list(set(routes[block]))[0]

        # Get the index(es) of the filter(s)
        filters = get_filter_index(model, block, filter, routes)

        for filter in filters:

            # Get information from the convolutional layer that is output of the FeatureConcat
            hyperparameters, parameters = get_layer_info(model.module_list[dest][0])

            # Creates a replica of the convolutional layer to perform pruning
            pruned_conv_layer = torch.nn.Conv2d(in_channels = hyperparameters['in_channels']-1,
                                                out_channels = hyperparameters['out_channels'],
                                                kernel_size = hyperparameters['kernel_size'],
                                                stride = hyperparameters['stride'],
                                                padding = hyperparameters['padding'],
                                                bias = False if parameters['bias'] is None else True                                  
                                                )
            
            
            # Removes convolutional filter
            parameters = remove_filter(parameters, filter, name = 'weight', channels = 'input')

            # Updates pruned convolutional layer
            pruned_conv_layer.weight.data = parameters['weight'].data
            pruned_conv_layer.weight.requires_grad = True

            # Exchanges the original layer with the pruned layer
            model = replace_layer(model, dest, pruned_conv_layer)

    # Removes convolutional filter from attribute related to .cfg file
    model.module_defs[block]['filters'] -= 1

    # Deletes auxiliary layers
    del pruned_conv_layer
    del pruned_batchnorm_layer

    log.write('Convolutional filter %d pruned from block %d\n' % (filter, block))

    return model

def norm(model, order = 'L2'):

    """ Computes the importance of convolutional filters based on the norm of the weights. """

    if order.upper() == 'L0':
      p = 0
    elif order.upper() == 'L1':
      p = 1
    elif order.upper() == 'L2':
      p = 2
    elif order.upper() == 'L-INF':
      p = float('inf')
    else:
      raise AssertionError('The order %s does not exist. Try L0, L1, L2 or L-Inf.' % (order)) 

    importances = list()

    blocks = to_prune(model)

    for block in blocks:

        n_filters = model.module_list[block][0].weight.data.shape[0]

        for i in range(n_filters):

            importance = model.module_list[block][0].weight[i].data.norm(p).item()
            importances.append([block, i, importance])

    return importances

def compute_vip(model):

    """ Calculates Variable Importance in Projection (VIP) from PLSRegression model. 
        (https://github.com/scikit-learn/scikit-learn/issues/7050) """

    # Matrices
    W = model.x_weights_
    T = model.x_scores_
    Q = model.y_loadings_

    # Number of features and number of components
    p, c = W.shape
    # Number of observations
    n, _ = T.shape

    # Variable Importance in Projection (VIP)
    VIP = np.zeros((p,))
    S = np.diag(T.T @ T @ Q.T @ Q).reshape(c, -1)
    S_cum = np.sum(S)
    for i in range(p):
        weight = np.array([(W[i,j] / np.linalg.norm(W[:,j]))**2 for j in range(c)])
        VIP[i] = np.sqrt(p*(S.T @ weight)/S_cum)

    return VIP

def pls_vip_single(model, X, Y, c):

    """ Single projection scheme. A single PLS model is learned considering all filters that compose the network at once.
        Based on paper Deep Network Compression Based on Partial Least Squares (https://www.sciencedirect.com/science/article/abs/pii/S0925231220305762) """

    # Project high dimensional space onto a low dimensional space (latent space)
    PLS = PLSRegression(n_components = c, scale = True)
    PLS.fit(X.T, Y)

    # Variable Importance in Projection (VIP) for each feature
    VIP = compute_vip(PLS)

    # Filters per layer
    blocks = to_prune(model)
    n_filters = list()
    for block in blocks:
        n_filters.append(int(model.module_list[block][0].out_channels))

    # Importances per layer
    VIPs = list()
    # Struct (Block/Filter/Importance)
    importances = list()

    for block in range(len(blocks)):

        start = sum(n_filters[:block])
        end = sum(n_filters[:block+1])
        VIPs.append(VIP[start:end])

        for filter in range(len(VIPs[block])):
            importances.append([blocks[block], filter, VIPs[block][filter]])

    return importances

def pls_vip_multi(model, X, Y, c):

    """ Multiple projections scheme. In this strategy, one PLS model is learned considering filters layer-by-layer. 
        Based on paper Deep Network Compression Based on Partial Least Squares (https://www.sciencedirect.com/science/article/abs/pii/S0925231220305762)""" 

    # Filters per layer
    blocks = to_prune(model)
    n_filters = list()
    for block in blocks:
        n_filters.append(int(model.module_list[block][0].out_channels))

    # Struct (Block/Filter/Importance)
    importances = list()

    for block in range(len(blocks)):

        # Separating input variable X by layer
        start = sum(n_filters[:block])
        end = sum(n_filters[:block+1])
        X_l = X[start:end]

        # Project high dimensional space onto a low dimensional space (latent space)
        PLS = PLSRegression(n_components = c, scale = True)
        PLS.fit(X_l.T, Y)

        # Variable Importance in Projection (VIP) for each feature of the current layer
        VIP_l = compute_vip(PLS)

        # Concatenating (Block/Filter/VIP)
        importances.append(np.column_stack([[blocks[block]]*n_filters[block], np.arange(n_filters[block]), VIP_l]))

        # Deleting current PLS model
        del PLS

    # Converting to list
    importances = pd.DataFrame(np.vstack(importances))
    importances[[0, 1]] = importances[[0, 1]].astype(int)
    importances = importances.to_records(index=False).tolist()

    return importances

def cca_multi(model, X, Y, c = 1):

    """ Multiple projections scheme. In this strategy, one CCA model is learned considering filters layer-by-layer. """

    # Filters per layer
    blocks = to_prune(model)
    n_filters = list()
    for block in blocks:
        n_filters.append(int(model.module_list[block][0].out_channels))

    # Struct (Block/Filter/Importance)
    importances = list()

    for block in range(len(blocks)):

        # Separating input variable X by layer
        start = sum(n_filters[:block])
        end = sum(n_filters[:block+1])
        X_l = X[start:end]

        # Project high dimensional space onto a low dimensional space (latent space)
        cca = CCA(n_components = c, scale = True, max_iter = 1)
        cca.fit(X_l.T, Y)

        # Projection matrix of the current layer
        W_l = cca.x_weights_

        # Concatenating (Block/Filter/Importance)
        importances.append(np.column_stack([[blocks[block]]*n_filters[block], np.arange(n_filters[block]), abs(W_l)]))

        # Deleting current CCA model
        del cca

    # Converting to list
    importances = pd.DataFrame(np.vstack(importances))
    importances[[0, 1]] = importances[[0, 1]].astype(int)
    importances = importances.to_records(index=False).tolist()

    return importances

def pls_lc_multi(model, X, Y):

    """ Multiple projections scheme. In this strategy, one PLS model is learned considering filters layer-by-layer. 
        Each projection matrix is linearly combined with convolutional filters to generate new ones. """
    
    # Filters per layer
    blocks = to_prune(model)
    n_filters = list()
    for block in blocks:
        n_filters.append(int(model.module_list[block][0].out_channels))

    # Struct (Block/Filter/Importance)
    importances = list()

    for i, block in enumerate(blocks):

        # Separating input variable X by layer
        start = sum(n_filters[:i])
        end = sum(n_filters[:i+1])
        X_l = X[start:end]

        # Number of components
        c = X_l.shape[0]

        # Project high dimensional space onto a low dimensional space (latent space)
        PLS = PLSRegression(n_components = c, scale = True)
        PLS.fit(X_l.T, Y)

        # Filters of current layer
        dim = model.module_list[block][0].weight.data.shape
        filters = torch.zeros(dim)
        # Dimension of weights
        size = (dim[1], dim[2], dim[3])

        print(i, PLS.x_weights_.shape)

        # For each column of projection matrix
        for j in range(PLS.x_weights_.shape[1]):
            # Broadcasting
            weights = [np.full(shape = size, fill_value = value) for value in PLS.x_weights_[:,j]]
            # Stack (n_filters, n_output, kernel_size, kernel_size)
            weights = torch.Tensor(np.stack(weights, axis = 0))
            # Element-wise tensor multiplication
            prod = torch.mul(weights, model.module_list[block][0].weight.data)
            # Element-wise sum (n_output, kernel_size, kernel_size)
            LC = torch.sum(prod, dim = 0)
            filters[j] = LC

        # Concatenating (Block/Filter/Importance)
        importances.append(np.column_stack([[block]*n_filters[i], np.arange(n_filters[i]), np.flip(np.arange(n_filters[i]))]))

        # Deleting current PLS model
        del PLS

        # Replace filters for linear combinations of the filters with the projection matrix
        with torch.no_grad():
            model.module_list[block][0].weight = torch.nn.Parameter(filters)

    # Converting to list
    importances = pd.DataFrame(np.vstack(importances))
    importances[[0, 1]] = importances[[0, 1]].astype(int)
    importances = importances.to_records(index=False).tolist()

    return importances

def split_pruned_filters(n_filters, iter):

    n_filters_iter = np.zeros((iter, len(n_filters)))

    for i, num in enumerate(n_filters):
        n_filters_iter[:,i] = np.array([num // iter + (1 if x < num % iter else 0)  for x in range (iter)])
    
    return n_filters_iter


def per_layer(model, rate):

    """ Calculates the number of filters that will be removed in each layer. """

    n_filters = list()

    blocks = to_prune(model)
    for block in blocks:
        n_filters.append(int(model.module_list[block][0].out_channels*rate))
        
    return n_filters

def select_filters(model, rate, importances, mode = 'layer', ascending = True, n_filters = None):

    """ Select the filters to be removed based on their respective importance. """ 

    # Importances as a dataframe
    importances = pd.DataFrame(importances, columns = ['Block', 'Filter', 'Importance'])
    # Sorting importances
    importances = importances.sort_values(by = 'Importance', ascending = ascending)    

    # Selected filters
    selected = list()

    if mode == 'layer':

        # Single pruning
        if n_filters is None:
            # Number of filters per layer to be removed
            n_filters = per_layer(model, rate = rate)

        # Selecting the filters for each layer that will be pruned
        blocks = list(importances['Block'].drop_duplicates().sort_values(ascending = True))

        if len(blocks) != len(n_filters):
            raise AssertionError('%d != %d\n' % (len(blocks), len(n_filters)))
        for i in range(len(blocks)):
            selected.append(importances.query('Block == @blocks[@i]')[:n_filters[i]].sort_values(by = 'Filter', ascending = False))
        selected = pd.concat(selected)

    else:
        
        # Total number of filters to be removed
        total_filters = int(len(importances)*rate)
        selected = importances.head(total_filters).sort_values(by = 'Filter', ascending = False)

        # Checking if all filters at some layer were selected to be removed
        will_be_pruned = dict(selected['Block'].value_counts())
        # Possible pruning blocks
        blocks = to_prune(model)
        # Number of filters per layer
        n_filters = per_layer(model, rate = 1.0)
        # Blocks that will have all filters pruned
        ignored = list()

        for key, value in will_be_pruned.items():
            if n_filters[blocks.index(key)] == will_be_pruned[key]:
                print('Warning: All filters at block [{}] were selected to be removed'.format(key))
                ignored.append(key)

        # Blocks that should have all filters pruned will not be pruned anymore
        selected = selected.query('Block not in @ignored')

    # Returns tuple with less important filters
    return list(selected.to_records(index=False))

def criteria_based_pruning(model, rate, rank, n_filters = None):

    """ Criteria-based pruning of convolutional filters in the model. """
  
    print('Criteria-based pruning method: %s.' % (rank.upper()))

    if rank.upper() in ['L0', 'L1', 'L2', 'L-INF']:
        importances = norm(model, order = rank)
        selected = select_filters(model, rate, importances, mode = 'layer', ascending = True, n_filters = n_filters)
    else:
        raise AssertionError('The rank %s does not exist. Try L0, L1, L2 or L-Inf.' % (rank))

    # Progress bar
    pbar = tqdm(total = len(selected), desc = 'Pruning convolutional filters')

    for i in range(len(selected)):
        block, filter, importance = selected[i]
        model = single_pruning(model, block, filter)
        # Update progress bar
        pbar.update(1)

    # Close progress bar
    pbar.close()

    print('%d filters were pruned.\n' % (len(selected)))

    return model

def projection_based_pruning(model, rate, technique, X, Y, c, n_filters = None):

    """ Projections-based pruning of convolutional filters in the model. """
  
    print('Projection-based pruning method: %s.' % (technique.upper()))

    if technique.upper() == 'PLS-VIP-SINGLE':
        importances = pls_vip_single(model, X, Y, c)
        selected = select_filters(model, rate, importances, mode = 'network', ascending = True)
    elif technique.upper() == 'PLS-VIP-MULTI':
        importances = pls_vip_multi(model, X, Y, c)
        selected = select_filters(model, rate, importances, mode = 'layer', ascending = True, n_filters = n_filters)
    elif technique.upper() == "CCA-CV-MULTI":
        importances = cca_multi(model, X, Y, c)
        selected = select_filters(model, rate, importances, mode = 'layer', ascending = True, n_filters = n_filters)
    elif technique.upper() == "PLS-LC-MULTI":
        importances = pls_lc_multi(model, X, Y)
        selected = select_filters(model, rate, importances, mode = 'layer', ascending = True, n_filters = n_filters)
    else:
        raise AssertionError('The technique %s does not exist. Try PLS-VIP-Single, PLS-VIP-Multi, CCA-Multi or PLS-LC-Multi.' % (technique))

    # Progress bar
    pbar = tqdm(total = len(selected), desc = 'Pruning convolutional filters')

    for i in range(len(selected)):
        block, filter, importance = selected[i]
        model = single_pruning(model, block, filter)
        # Update progress bar
        pbar.update(1)

    # Close progress bar
    pbar.close()

    print('%d filters were pruned.\n' % (len(selected)))

    return model

def correlation_matrices(model, measure):

    """ Compute correlation coefficient matrices. """

    # Convolutional blocks to prune
    blocks = to_prune(model)

    # Correlation coefficient matrices
    CCM = list()

    # Progress bar
    pbar = tqdm(total = len(blocks), desc = 'Computing correlation matrix')

    # Loop over convolutional layers
    for l, block in enumerate(blocks):

        n_filters = model.module_list[block][0].weight.shape[0]
        # One correlation coefficient matrix per layer
        CCM.append(np.zeros((n_filters, n_filters)))

        # Loop over convolutional filters of the current layer
        for i in range(n_filters):
            fi = model.module_list[block][0].weight[i].data
            for j in range(n_filters):
                fj = model.module_list[block][0].weight[j].data
                # Pearson’s Correlation (linear relationship)
                if measure.lower() == 'pearson':
                    CCM[l][i][j] = pearsonr(fi.flatten().detach().numpy(), fj.flatten().detach().numpy())[0]
                # Spearman's Correlation (non-linear relationship)
                elif measure.lower() == 'spearman':
                    CCM[l][i][j] = spearmanr(fi.flatten().detach().numpy(), fj.flatten().detach().numpy())[0]
                # Kendall's Correlation (non-linear relationship)
                elif measure.lower() == 'kendall':
                    CCM[l][i][j] = kendalltau(fi.flatten().detach().numpy(), fj.flatten().detach().numpy())[0]
                else:
                    raise AssertionError('The measure %s does not exist. Try Pearson, Spearman or Kendall.' % (measure))

        # Update progress bar
        pbar.update(1)
    
    # Close progress bar
    pbar.close()

    return CCM

def cluster_analysis(CCM, clustering, block):

    """ Selects filters of a given block for removal through a cluster analysis. """

    # Correlations of filter fi with all filters of the other clusters
    corr = list()
    # Maximum correlation of each filter in cluster ci
    max_corr = list()
    # Filters selected for removal
    selected = list()

    # Dictionary containing the clusters with their respective filters
    clusters = collections.defaultdict(list)
    for fi, ci in enumerate(clustering.labels_):
        clusters[ci].append(fi)

    # Clusters with more than one filter
    more_than_one = {x: count for x, count in collections.Counter(clustering.labels_).items() if count > 1}

    # For each cluster that has more than one filter
    for ci, nf in sorted(more_than_one.items()):

        # For each filter in that cluster
        for fi in clusters[ci]:

            # Compare the correlation of this filter with all filters in each existing cluster
            for cj in sorted(clusters.keys()):
                # Do not compare filters from the same cluster
                if ci != cj:
                    # Correlations between filter fi and filters of cluster cj
                    corr.extend(CCM[block][fi][clusters[cj]])

            # Maximum correlation of the filter fi with the other clusters
            max_corr.append(np.max(corr))
            # Clearing the correlations of the filter fi with all filters of the other clusters
            corr.clear()

        # Selects the (nf-1) filters in cluster ci that have the highest correlation with the other clusters
        arg_selected = np.argsort(max_corr)[-(nf-1):]
        selected.extend(np.take(clusters[ci], arg_selected))
        # Clearing the maximum correlations of the filters in cluster ci
        max_corr.clear()

    return selected

def agglomerative_clustering(model, rate, CCM, n_filters = None):

    """ Agglomerative clustering pruning method based on correlation between convolutional filters.
    Based on paper Deep Network Pruning for Object Detection (https://ieeexplore.ieee.org/document/8803505) """

    # Prunable blocks
    blocks = to_prune(model)

    # Selected filters
    selected = list()

    # Progress bar
    pbar = tqdm(total = len(blocks), desc = 'Selecting filters to prune')

    for i, block in enumerate(blocks):

        # Number of filters in the current layer
        f_layer = CCM[i].shape[0]

        # Number of filters to prune
        if n_filters is None:
            n_prunable = int(rate*f_layer)
        else:
            n_prunable = n_filters[i]

        # Clustering filters of the current layer
        clustering = AgglomerativeClustering(n_clusters = (f_layer - n_prunable), 
                                             linkage = 'complete', 
                                             affinity = 'precomputed').fit(1-CCM[i])
        # Select filters of the current layer to prune
        selected_l = cluster_analysis(np.absolute(CCM), clustering, i)

        # Concatenating (Block/Filter/Importance)
        selected.append(np.column_stack([[block]*n_prunable, sorted(selected_l, reverse = True)]))

        # Update progress bar
        pbar.update(1)

    # Close progress bar
    pbar.close()

    # Converting to list
    selected = pd.DataFrame(np.vstack(selected))
    selected = selected.astype(int)
    selected = selected.to_records(index=False).tolist()

    return selected

def wrapper_based_pruning(model, rate, technique, CCM = None, n_filters = None):

    """ Wrapper approaches include a classification/clustering algorithm in the filter evaluation step. """

    print('Wrapper-based pruning method: %s.' % (technique.upper()))

    if technique.upper() == 'HAC':
        selected = agglomerative_clustering(model, rate, CCM, n_filters)
    else:
        raise AssertionError('The technique %s does not exist. Try HAC.' % (technique))
    
    # Progress bar
    pbar = tqdm(total = len(selected), desc = 'Pruning convolutional filters')

    for i in range(len(selected)):
        block, filter = selected[i]
        model = single_pruning(model, block, filter)
        # Update progress bar
        pbar.update(1)

    # Close progress bar
    pbar.close()

    print('%d filters were pruned.\n' % (len(selected)))

    return model

def random_pruning(model, rate, seed = 42, n_filters = None):

    """ Random pruning of convolutional filters in the model. """

    if seed != -1:
        print('Random pruning with seed %d\n' % (seed))
    else:
        print('Random pruning without seed\n')

    blocks = to_prune(model)

    # Single pruning
    if n_filters is None:
        # Number of filters per layer to be removed
        n_filters = per_layer(model, rate)

    if len(blocks) != len(n_filters):
        raise AssertionError('%d != %d\n' % (len(blocks), len(n_filters)))

    # Progress bar
    pbar = tqdm(total = sum(n_filters), desc = 'Pruning convolutional filters')

    for i in range(len(blocks)):

        if seed != -1:
            random.seed(seed)
        filters = -np.sort(-np.array(random.sample(range(model.module_list[blocks[i]][0].out_channels), n_filters[i])))

        for filter in filters:
            model = single_pruning(model, blocks[i], filter)
            # Update progress bar
            pbar.update(1)

    print('%d filters were pruned.' % (sum(n_filters)))

    # Close progress bar
    pbar.close()

    return model

def feature_extraction(conv_map, pool_type = 'max'):

    """ Represents the output of the filters that compose the network as feature vectors for a single image. """

    # Features
    features = list()

    # For each convolutional layer
    for l in range(len(conv_map)):

        # For each batch
        for b in range(len(conv_map[l])):
    
            n_filters = len(conv_map[l][b])

            # For each filter
            for f in range(n_filters):

                # Global Average Pooling
                if pool_type.lower() == 'avg':
                    global_pool = torch.nn.AvgPool2d(kernel_size = conv_map[l][b][f].shape)
                # Global Max Pooling
                else:
                    global_pool = torch.nn.MaxPool2d(kernel_size = conv_map[l][b][f].shape)
                
                feature = global_pool(conv_map[l][b][f].unsqueeze(0))
                features.append(float(feature))

    return features

def get_variables(model, data, img_size, num_classes, pool_type, perc_samples):

    """ Extracts feature maps and transform them into feature vectors for projection """

    # Output filename
    filename = 'variables_' + pool_type + '.npy'
    # Open file with wb mode
    f = open(filename, 'wb')

    # Extracts all feature maps from all layers of the network for each image in the dataset
    inputs, labels, img_sizes = filter_representation(model = model, 
                                                      data = data, 
                                                      img_size = img_size, 
                                                      pool_type = pool_type,
                                                      subset = 'train',
                                                      route = False,
                                                      perc_samples = perc_samples)

    # Reshape input variables (filters x images)
    X = np.array(inputs).reshape((len(inputs[0]), len(inputs)))

    # Saving matrix X
    np.save(f, X)

    print('Number of images:', X.shape[1])
    print('Number of filters per image:', X.shape[0])

    print('Shape of input variables:', X.shape)

    # Computes the class label matrix of the training data
    Y = class_label_matrix(labels, img_sizes, num_classes = num_classes)
    print('Shape of output variables:', Y.shape)

    # Saving matrix Y
    np.save(f, Y)

    # Close file
    f.close()

    return X, Y

def filter_representation(model, data, img_size, pool_type = 'max', subset = 'train', route = False, perc_samples = 1.0):

    """ Extract features from all convolutional maps for each image in the subset. """

    # Initializing activation map lists
    inputs = list()
    conv_i = list()
    out_i = list()
    yolo_out_i = list()

    # Image sizes
    img_sizes = list()

    # Initializing the model
    device = torch_utils.select_device()
    print('Device:', device)
    model = model.to(device)

    # Prunable block indexes
    blocks = to_prune(model)

    # Load dataset images
    data = parse_data_cfg(data)
    path = data[subset]
    dataset = LoadImagesAndLabels(path = path, img_size = img_size, rect = True, single_cls = False)

    # Number of images to extract the activation map
    num_images = int(perc_samples*(len(dataset)))

    # Get convolutional feature maps for each image
    for i in tqdm(range(num_images), desc = 'Extracting activation maps per image'):

        # Image pre-processing
        img0, _, _ = load_image(dataset, i)
        img = letterbox(img0, new_shape=img_size)[0]
        x = torch.from_numpy(img)

        # Append image size to list
        img_sizes.append(img.shape[:2])
        
        # Normalization
        x = x.to(device).float() / 255.0
        x = x.float() / 255.0
        x = x.unsqueeze(1)
        x = x.permute(1, 3, 2, 0)

        # For each layer
        for j, module in enumerate(model.module_list):

            name = module.__class__.__name__

            # Sum with WeightedFeatureFusion() and concat with FeatureConcat()
            if name in ['WeightedFeatureFusion', 'FeatureConcat']:  
                x = module(x, out_i)
            elif name == 'YOLOLayer':
                yolo_out_i.append(module(x, out_i))
            # Run module directly, i.e. mtype = 'convolutional', 'upsample', 'maxpool', 'batchnorm2d' etc
            else:

                if name == 'Sequential':

                    # Block (Conv2D + BatchNorm2D + LeakyReLU)
                    if len(module) > 1:

                        # Convolution
                        Conv2D = module[0]
                        x = Conv2D(x)
                        if j in blocks:
                           conv_i.append(x)
                        
                        # Batch normalization
                        BatchNorm2D = module[1]
                        x = BatchNorm2D(x)
                        
                        # Activation
                        LeakyReLU = module[2]
                        x = LeakyReLU(x)

                    else:
                        # Single Conv2D
                        x = module(x)
                        if j in blocks:
                            conv_i.append(x)

                # Upsample
                else:
                    x = module(x)

            if route is True:
                out_i.append(x if model.routs[j] else [])
            else:
                out_i.append(x)

        # Feature extraction
        features = feature_extraction(conv_i, pool_type = pool_type)
        inputs.append(features)
            
        # Clearing GPU Memory
        conv_i.clear()
        out_i.clear()
        yolo_out_i.clear()
        del x, img0, img

    return inputs, dataset.labels[:num_images], img_sizes

def class_label_matrix(labels, img_sizes, num_classes):

    """ Computes the class label matrix of the training data. """

    # Class label matrix
    Y = list()

    # Modeling the object detection problem as a binary classification problem (none, detection)
    if num_classes == 2:

        print('Modeling as a binary problem')

        for sample in range(len(labels)):
            # None
            if len(labels[sample]) == 0:
                Y.append(0)
            # Detection (smoke or fire or both)
            else:
                Y.append(1)

    # Modeling the object detection problem as a multiclass classification problem (none, fire, smoke)
    if num_classes > 2:

        print('Modeling as a multiclass problem')

        # Pixels area per image
        area = {'fire': 0, 'smoke': 0}

        for sample in range(len(labels)):
            # None
            if len(labels[sample]) == 0:
                Y.append(0)

            # Detection
            else:
                # For each bounding box
                for label in range(labels[sample].shape[0]):
                    
                    # Class identifier
                    class_id = labels[sample][label][0]

                    # Normalized coordinates
                    xmin = labels[sample][label][1]
                    ymin = labels[sample][label][2]
                    xmax = labels[sample][label][3]
                    ymax = labels[sample][label][4]

                    # Image dimensions
                    height = img_sizes[sample][0]
                    width = img_sizes[sample][1]

                    # Coordinates without normalization                 
                    xmin, ymin, xmax, ymax = deconvert((width, height), (xmin, ymin, xmax, ymax))

                    # Sum the pixel areas according to the class
                    if class_id == 0:
                        area['smoke'] += (xmax - xmin) * (ymax - ymin)
                    else:
                        area['fire'] += (xmax - xmin) * (ymax - ymin)

                # If the smoke pixel area is larger than the fire pixel area
                if area['smoke'] > area['fire']:
                    Y.append(1)
                # Otherwise
                else:
                    Y.append(2)

                # Resetting counters for the next image
                area = area.fromkeys(area, 0)
        
        # Convert a class vector (integers) to binary class matrix
        Y = np.eye(num_classes, dtype = 'int')[Y]
    
    # List to numpy array
    Y = np.array(Y)

    return Y