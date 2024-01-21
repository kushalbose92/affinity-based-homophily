"""
program: utilities needed by affinity matrix computation script
author: indranil ojha

"""
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.random import set_seed as tf_seed
from torch_geometric.utils import remove_isolated_nodes, homophily 
from numpy.random import seed as np_seed
from random import seed as random_seed
import numpy as np
import os

# set all seeds for reproducibility
def set_seeds(seed=13):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random_seed(seed)
    tf_seed(seed)
    np_seed(seed)

# retrieve best parameters for dataset
def get_params(dataset_name, root):
    params_fname = f"{root}params/params.csv"
    params_df = pd.read_csv(params_fname)
    params_df = params_df[params_df['dataset']==dataset_name]
    params_df = params_df[['emb_ratio', 'max_nodes', 'k', 'init_lr', 'lr_decay', 'epochs']]
    emb_ratio, max_nodes, k, init_lr, lr_decay, epochs = params_df.iloc[0].values
    max_nodes = int(max_nodes)
    epochs = int(epochs)
    return(emb_ratio, max_nodes, k, init_lr, lr_decay, epochs)
    
# retrieve best parameters for dataset
def get_edge_h(dataset_name, root):
    edge_h_fname = f"{root}params/edge_h.csv"
    edge_h_df = pd.read_csv(edge_h_fname)
    edge_h_df = edge_h_df[edge_h_df['dataset']==dataset_name]
    edge_h_df = edge_h_df['edge_h']
    return(edge_h_df.iloc[0])

def get_degree_1D(edge_index, n_nodes):
    deg = np.zeros(n_nodes)
    for i in range(n_nodes):
        deg[i] = np.sum(edge_index.numpy()[0,:]==i)
    return(deg)

def remove_isolated(data):
    n_nodes_before =data.x.shape[0]
    _, _, mask = remove_isolated_nodes(data.edge_index, num_nodes=n_nodes_before)
    data = data.subgraph(mask)
    n_nodes_after =data.x.shape[0]
    nodes_removed = n_nodes_before - n_nodes_after
    return(data, nodes_removed)

def get_homophily(data):
    h = homophily(data.edge_index, data.y, method='edge')
    return(h)

def plot_hist(hist_loss, hist_aff, dataset, results_folder):
    # summarize trend for loss & affinity score
    plt.subplot(121)
    plt.plot(hist_loss)
    #plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.subplot(122)
    plt.plot(hist_aff)
    #plt.title('affinity score')
    plt.ylabel('affinity score')
    plt.xlabel('epoch')
    plt.ylim(0,1)
    plt.suptitle(dataset)
    savefile=f"{results_folder}/hist/{dataset}.png"
    plt.savefig(savefile)
    plt.close()
