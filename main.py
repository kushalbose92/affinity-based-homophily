"""
program: compute affinity matrix for dataset
version: sparse version, single headed-affinity
author: indranil ojha

"""

# setup environment
root = 'D:/Indranil/ML2/projects/TinyGCN_research/affinity_baseline/'
data_folder = "D:/Indranil/ML2/Datasets/"

# import libraries, including utils
import os, sys
#root = os.getcwd().replace('\\','/')+'/'
sys.path.append(root)

import tensorflow as tf
import pandas as pd
import numpy as np
from torch_geometric.loader import RandomNodeLoader
import time
from utils import set_seeds, get_params, get_edge_h, remove_isolated, get_homophily, plot_hist
from models import get_affinity_model
from load import load_dataset
    
###
### main program ###
###

seed = 13
set_seeds(seed)

results_folder = f"{root}results/"
if not os.path.exists(results_folder):
    os.makedirs(results_folder)
if not os.path.exists(results_folder+'hist'):
    os.makedirs(results_folder+'hist')
if not os.path.exists(results_folder+'aff_plots'):
    os.makedirs(results_folder+'aff_plots')

datasets = ['Texas', 'Cornell', 'Wisconsin', 'Cora', 'CiteSeer','Squirrel', 'Chameleon', 'PubMed', 'ogbn-arxiv']
datasets = ['PubMed']

results = []
result_cols = ['dataset', 'nodes', 'features', 'classes', 'directed', 'emb_ratio', 
               'max_nodes', 'k', 'init_lr', 'lr_decay', 'epochs', 'edge_h', 
               'affinity_h', 'elapsed', 'seed']

try:
   results_df = pd.read_csv(results_folder+'results.csv') 
except:
    results_df = pd.DataFrame(columns = result_cols)

#results_df = pd.DataFrame(columns = result_cols)
problem_datasets = []

print()
for dataset_name in datasets:

    print()
    # load entire dataset
    data, n_classes = load_dataset(data_folder, dataset_name, largest_connected_components=False)
    n_nodes, n_features = data.x.shape
    n_edges = data.edge_index.shape[1]
    is_directed = data.is_directed()
    
    # read edge homophily value for dataset, for reporting
    edge_h = get_edge_h(dataset_name, root)    

    # get best parameters
    emb_ratio, max_nodes, k, init_lr, lr_decay, epochs = get_params(dataset_name, root)
    #emb_ratio, max_nodes, k, init_lr, lr_decay, epochs = 2, 5000, 50, 0.001, 0.96, 500
    n_emb_features = int(n_features/emb_ratio)
    epochs = int(epochs)

    print(f'{dataset_name:<12} #nodes: {n_nodes:>5} #features: {n_features:>5} #classes: {n_classes:>2} #edges: {n_edges:>5} Directed: {is_directed}')
    start = time.time()

    # create iterator for splits of manageable sizes
    n_rounds = n_nodes//max_nodes + 1 if max_nodes<n_nodes else 1 # no of rounds needed rto cover all nodes
    loader = RandomNodeLoader(data, num_parts=n_rounds, shuffle=True)
    h_list, aff_list = [], []
    
    # loop thru each round
    cnt = 1
    for data in loader:
        print(f"{cnt}/{n_rounds}. ", end="")
        cnt += 1
        data, _ = remove_isolated(data)
        h = get_homophily(data)
        X, y, edge_index = data.x, data.y, data.edge_index
        X, y = tf.convert_to_tensor(X.numpy(), dtype=tf.float32), tf.convert_to_tensor(y.numpy())
        n_nodes, n_features = data.x.shape
        n_edges = data.edge_index.shape[1]
        edge_index = tf.cast(edge_index, tf.int64)
        edge_index = tf.sparse.SparseTensor(indices=tf.transpose(edge_index), values=tf.ones(n_edges), dense_shape=[n_nodes, n_nodes])
        # compute affinity based homophily score
        try:
            # create & train affinity model
            affinity_model = get_affinity_model(dataset_name, n_nodes, n_features, 
                                    n_edges, n_emb_features, is_directed, k)
            optim = tf.keras.optimizers.Adam(learning_rate=init_lr)
            hist_loss, hist_aff = affinity_model.train(X, edge_index, epochs, optim, results_folder)
            
            # plot affinity matrix against adjacency
            affinity_model.plot_affinity(X, edge_index, save_flag=True, results_folder=results_folder)
    
            # compute affinity score
            affinity_h = affinity_model.get_affinity_score(X)
            #_, _ = affinity_model.get_params()
            print(f"affinity-homophily: {affinity_h:.4f}, edge-homophily: {edge_h:.4f}")
            plot_hist(hist_loss, hist_aff, dataset_name, results_folder)

            # store values for averaging
            h_list.append(h)
            aff_list.append(affinity_h)
            
            #affinity_model.save(f'{dataset_name}_model.h5')
    
        except:
            print('... cannot compute affinity score')
            problem_datasets.append([dataset_name, 'cannot compute affinity score'])
            #datasets.remove(dataset_name)
            continue
    
    affinity_h = np.mean(aff_list)
    print(f"final homophily: affinity:{affinity_h:.4f}, edge={edge_h:.4f}")

    stop = time.time()
    elapsed = int(stop-start)
    results_df.loc[len(results_df)] = [dataset_name, n_nodes, n_features, n_classes, is_directed, 
            emb_ratio, max_nodes, k, init_lr, lr_decay, epochs, edge_h, affinity_h, elapsed, seed]

    # save results
    results_df.to_csv(results_folder+'results.csv', index=False)
    del affinity_model
        
# report logs
if len(problem_datasets)>0:
    print("\nProblem encountered with following datasets:")
    for name, reason in problem_datasets:
        print(f"{name:<20} {reason:<50}")
