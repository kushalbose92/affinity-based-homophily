"""
program:
    compare edge based vs afinitty based homophily 
    tensorflow version 
    single-headed

"""

# setup environment
env = 'laptop' # laptop / colab / ecsu

if env=='colab':
    from google.colab import drive
    drive.mount("/content/gdrive")

if env=='colab':
    root = "/content/gdrive/MyDrive/Colab Notebooks/ICLR2024_tiny/"
    data_folder = "/content/gdrive/MyDrive/Colab Notebooks/Datasets/"
elif env=='laptop':
    root = "D:/Indranil/ML2/projects/GCN/affinity/ICLR2024_tiny/"
    data_folder = "D:/Indranil/ML2/Datasets/"
elif env=='ecsu':
    root = "/home/iplab/indro/ml2/ICLR2024_tiny/"
    data_folder = "/home/iplab/indro/ml2/Datasets/"

# import libraries, including utils
import os
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kendalltau
#import winsound
import sys
sys.path.append(root)
from utils_tf import load_data, compute_F1, rbo
import time

# define functions and models

# retrieve best parameters for dataset
def get_params(dataset_name, root):
    params_fname = f"{root}params.csv"
    params_df = pd.read_csv(params_fname)
    params_df = params_df[params_df['dataset']==dataset_name]
    params_df = params_df[['emb_ratio', 'k', 'init_lr', 'lr_decay', 'epochs']]
    return(params_df.iloc[0].values)
    
# retrieve best parameters for dataset
def get_edge_h(dataset_name, root):
    edge_h_fname = f"{root}edge_h.csv"
    edge_h_df = pd.read_csv(edge_h_fname)
    edge_h_df = edge_h_df[edge_h_df['dataset']==dataset_name]
    edge_h_df = edge_h_df['edge_h']
    return(edge_h_df.iloc[0])

# define affinity model
class get_affinity_model(tf.keras.Model):
    def __init__(self, n_nodes, n_features, n_emb_features, k=10):
        super().__init__()
        self.n_nodes = n_nodes
        self.n_features = n_features
        self.n_emb_features = n_emb_features
        self.inverted_eye = tf.ones((self.n_nodes,self.n_nodes))-tf.eye(self.n_nodes) # inverted diag matrix
        self.k = k
        self.eps = 0.000001
        self.W = self.add_weight(name='W',
            shape=[self.n_features, self.n_emb_features],
            initializer=tf.random_normal_initializer())
        self.Z = self.add_weight(name='Z',
            shape=[self.n_nodes, self.n_emb_features],
            initializer=tf.random_normal_initializer())

    def forward(self, X):
        #Z_emb = tf.linalg.matmul(self.Z, self.W)
        X_emb = tf.linalg.matmul(X, self.W)
        affinity = tf.linalg.matmul(self.Z, tf.transpose(X_emb)) 
        Z_norm = tf.linalg.norm(self.Z, axis=1)
        X_norm = tf.linalg.norm(X_emb, axis=1)
        affinity = affinity/(X_norm+self.eps)
        affinity = tf.transpose(tf.transpose(affinity)/(Z_norm+self.eps))
        
        if not is_directed:
            affinity_tranposed = tf.transpose(affinity)
            affinity = (affinity + affinity_tranposed)/2.0
        
        return(affinity)

    def backward(self, adj, affinity):
        # push values towards 0 or 1 - k determines power of sigmoid function to do this
        affinity = tf.keras.activations.sigmoid(affinity*self.k) 
    
        adj_new = tf.math.multiply(adj, self.inverted_eye)
        affinity_new = tf.math.multiply(affinity, self.inverted_eye)

        loss = 1 - compute_F1(adj_new, affinity_new)
        return(loss)

    def train(self, X, adj, epochs, optim):
        history = []
        print("Training affinity model ... ", end="")
        print("    ", end='')
        for e in range(epochs):
            print(f"\b\b\b\b{int(100*e/epochs):>3}%", end='')
            with tf.GradientTape() as tape:
                affinity = self.forward(X)
                loss = self.backward(adj, affinity)
            grads = tape.gradient(loss, self.trainable_weights)
            optim.apply_gradients(zip(grads, self.trainable_weights))           
            history.append(loss.numpy())
            if e%25==0:
                self.plot_affinity(X)
        print("\b\b\b\b... done.")
        return(history)

    def get_params(self):
        return(self.W, self.Z)

    def get_affinity_matrix(self, X, sigmoid_flag=False):
        affinity = self.forward(X)
        if sigmoid_flag:
            affinity = tf.keras.activations.sigmoid(affinity*self.k)
        else: # move to 0-1 range            
            affinity = (affinity+1)/2
        return(affinity)

    def get_affinity_score(self, X): 
        affinity = self.get_affinity_matrix(X)

        affinity = self.get_affinity_matrix(X, sigmoid_flag=True)
        all_sum = tf.reduce_sum(affinity).numpy()
        diag_sum = tf.reduce_sum(tf.linalg.diag_part(affinity)).numpy()
        non_diag_sum = all_sum - diag_sum
        diag_mean = diag_sum/self.n_nodes
        non_diag_mean = non_diag_sum/(self.n_nodes * (self.n_nodes - 1))
        diag_score = diag_mean/non_diag_mean
        normalized_diag_score = np.log(diag_score) /np.log((self.n_nodes * (self.n_nodes - 1))/n_edges)

        return(normalized_diag_score)

    def plot_affinity(self, X, save_flag=False): 
        # plot affinity matrix against adjacency
        affinity = self.get_affinity_matrix(X, sigmoid_flag=True)
        plt.figure(figsize=(7,4))
        plt.subplot(121)
        plt.imshow(adj[:100,:100], cmap='gray')
        plt.axis('off')
        plt.subplot(122)
        plt.imshow(affinity[:100,:100], cmap='gray')
        plt.axis('off')
        plt.suptitle(f"Dataset - {dataset_name}", fontsize=20)
        plt.tight_layout()
        if save_flag:
            savefile=f"{folder}aff_plots/{dataset_name}.png"
            plt.savefig(savefile)
        plt.show()

def plot_hist(history, dataset):
    # summarize history for loss
    plt.plot(history)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title(dataset)
    savefile=f"{folder}hist/{dataset}.png"
    plt.savefig(savefile)
    plt.show()
    
###
### main program ###
###

seed = 13
np.random.seed(seed)

folder = f"{root}results_tf/"
if not os.path.exists(folder):
    os.makedirs(folder)
if not os.path.exists(folder+'hist'):
    os.makedirs(folder+'hist')
if not os.path.exists(folder+'aff_plots'):
    os.makedirs(folder+'aff_plots')

datasets = ['wisconsin']

results = []
result_cols = ['dataset', 'nodes', 'features', 'classes', 'directed', 'emb_ratio', 'k', 'init_lr', 'lr_decay', 'epochs', 'edge_h', 'affinity_h', 'elapsed']

try:
   results_df = pd.read_csv(folder+'results.csv') 
except:
    results_df = pd.DataFrame(columns = result_cols)

#results_df = pd.DataFrame(columns = result_cols)
problem_datasets = []

print()
for dataset_name in datasets:

    start = time.time()
    # load features and adjacency matrix
    X, y, adj_sparse, adj, _, _, n_classes, is_directed, error_status = load_data(
        dataset_name, remove_isolated=True)
    if error_status:
        print(f"dataset: {dataset_name} - ### Loading error ###")
        problem_datasets.append([dataset_name, 'loading error'])
        #datasets.remove(dataset_name)
        continue
    n_nodes, n_features = X.shape
    n_edges = int(np.sum(adj))
    print(f'{dataset_name:<12} #nodes: {n_nodes:>5} #features: {n_features:>5} #classes: {n_classes:>2} #edges: {n_edges:>5} Directed: {is_directed}')

    # read edge homophily value for dataset, for reporting
    edge_h = get_edge_h(dataset_name, root)    
    
    # compute affinity based homophily score
    try:
        # get best parameters
        emb_ratio, k, init_lr, lr_decay, epochs = get_params(dataset_name, root)
        epochs = int(epochs)

        # create & train affinity model
        n_emb_features = int(n_features/emb_ratio)
        affinity_model = get_affinity_model(n_nodes, n_features, n_emb_features, k=k)
        optim = tf.keras.optimizers.Adam(learning_rate=init_lr)
        history = affinity_model.train(X, adj, epochs=epochs, optim=optim)
        plot_hist(history, dataset_name)
        
        # plot affinity matrix against adjacency
        affinity_model.plot_affinity(X, save_flag=True)

        # compute affinity score
        affinity_h = affinity_model.get_affinity_score(X)
        print(f"affinity-homophily: {affinity_h:.2f}, edge-homophily: {edge_h:.2f}")

        #affinity_model.save(f'{dataset_name}_model.h5')

    except:
        print('... cannot compute affinity score')
        problem_datasets.append([dataset_name, 'cannot compute affinity score'])
        #datasets.remove(dataset_name)
        continue

    stop = time.time()
    elapsed = int(stop-start)
    results_df.loc[len(results_df)] = [dataset_name, n_nodes, n_features, n_classes, is_directed, 
            emb_ratio, k, init_lr, lr_decay, epochs, edge_h, affinity_h, elapsed]
    
# save results
results_df.to_csv(folder+'results.csv', index=False)

# report logs
if len(problem_datasets)>0:
    print("\nProblem encountered with following datasets:")
    for name, reason in problem_datasets:
        print(f"{name:<20} {reason:<50}")
