"""
program:
    retrieves homophily scores f different datasets
    compares and creates reports
    tensorflow version 

version:
    split from last working version (v1), then that was split into two scripts
    first script (compute_homophily):
        mainly trains model - configurablle hyperparameters
        computes edge-based and affinity-based homophily scores
        stores training models & parameters, and homophily scores
    second script (this one):
        retrerives scores for various daatsets, and creates consolidated reports
    there may be few more scripts that creates repots in this scrit but customized for paper
 
"""

# setup environment
env = 'laptop' # laptop / colab / ecsu
root = 'D:/Indranil/ML2/projects/TinyGCN_research/affinity_baseline/results/'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt    
###
### main program ###
###

# small datasets
fname = root+'results.csv'
results_df = pd.read_csv(fname)[['dataset', 'edge_h', 'affinity_h']]

results_df.sort_values(by='edge_h', axis=0, ascending=False, inplace=True)

if len(results_df)>1:
    # plot results
    plt.figure(figsize=(12,7), dpi=600)
    X_axis = np.arange(len(results_df))
    plt.bar(X_axis - 0.2, results_df['edge_h'].values, 0.3, label = 'Edge-homophily')
    plt.bar(X_axis + 0.2, results_df['affinity_h'].values, 0.3, label = 'Affinity-homophily')
    plt.xticks(X_axis, results_df['dataset'].values, rotation=45, fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylabel("Homophily / Affinity", fontsize=18)
    plt.ylim(0,1)
    plt.vlines(2.5, 0, 1, 'b','dotted')
    plt.text(0.05, 0.88, 'Homophily', fontsize = 18, color='b') 
    plt.text(4.5, 0.4, 'Heterophily', fontsize = 18, color='b') 
    plt.legend(fontsize=16)
    plt.savefig(f'{root}comparison.png')
    plt.show()
