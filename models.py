"""
program: model definition for affinity matrix computation
author: indranil ojha

"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# given true and predicted values of binary tensor
# compute TP, FP, TN, FN & F1-score
def compute_F1(y_true, y_pred):
    y_diff = tf.sparse.add(y_true, y_pred*(-1))
    #mismatches = tf.math.sqrt(tf.math.multiply(y_diff, y_diff))
    #mismatches = tf.math.multiply(y_diff, y_diff)
    mismatches = tf.abs(y_diff)
    matches = 1 - mismatches
    tp = tf.reduce_sum(tf.math.multiply(matches, y_pred))
    fp = tf.reduce_sum(tf.math.multiply(mismatches, y_pred))
    fn = tf.reduce_sum(tf.math.multiply(mismatches, 1 - y_pred))    

    precision = tp / (tp+fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    
    return(f1)   

# define affinity model
class get_affinity_model(tf.keras.Model):
    def __init__(self, dataset_name, n_nodes, n_features, 
                 n_edges, n_emb_features, is_directed, k=10):
        super().__init__()
        self.dataset_name = dataset_name
        self.n_nodes = n_nodes
        self.n_features = n_features
        self.n_edges = n_edges
        self.n_emb_features = n_emb_features
        self.is_directed = is_directed
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
        
        if not self.is_directed:
            affinity_tranposed = tf.transpose(affinity)
            affinity = (affinity + affinity_tranposed)/2.0
        
        return(affinity)

    def backward(self, edge_index, affinity):
        # push values towards 0 or 1 - k determines power of sigmoid function to do this
        affinity = tf.keras.activations.sigmoid(affinity*self.k) 
    
        edge_index_new = edge_index.__mul__(self.inverted_eye)
        affinity_new = tf.math.multiply(affinity, self.inverted_eye)

        loss = 1 - compute_F1(edge_index_new, affinity_new)
        return(loss)

    def train(self, X, edge_index, epochs, optim, results_folder):
        hist_loss, hist_aff = [], []
        print("Training affinity model ... ", end="")
        print("    ", end='')
        for e in range(epochs):
            print(f"\b\b\b\b{int(100*e/epochs):>3}%", end='')
            with tf.GradientTape() as tape:
                affinity = self.forward(X)
                loss = self.backward(edge_index, affinity)
            grads = tape.gradient(loss, self.trainable_weights)
            optim.apply_gradients(zip(grads, self.trainable_weights))           
            aff_score = self.get_affinity_score(X)
            hist_loss.append(loss.numpy())
            hist_aff.append(aff_score)

            #if e%50==0:
            #    self.plot_affinity(X, edge_index, results_folder)
        print("\b\b\b\b... done.")
        return(hist_loss, hist_aff)

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
        affinity = self.get_affinity_matrix(X, sigmoid_flag=True)
        all_sum = tf.reduce_sum(affinity).numpy()
        diag_sum = tf.reduce_sum(tf.linalg.diag_part(affinity)).numpy()
        non_diag_sum = all_sum - diag_sum
        diag_mean = diag_sum/self.n_nodes
        non_diag_mean = non_diag_sum/(self.n_nodes * (self.n_nodes - 1))
        diag_score = diag_mean/non_diag_mean
        normalized_diag_score = np.log(diag_score+1) /np.log((self.n_nodes * (self.n_nodes - 1))/self.n_edges+1)

        return(max(0,normalized_diag_score))

    def plot_affinity(self, X, edge_index, results_folder, save_flag=False, n_points=100): 
        # take first few nodes and crate dense tensor for image plotting
        def make_dense(e, n_points=100):
            idx0 = np.where(e.indices.numpy()[:,0]<n_points)[0]
            idx1 = np.where(e.indices.numpy()[:,1]<n_points)[0]
            idx = list(set(idx0).intersection(set(idx1)))
            e_indices = e.indices.numpy()[idx,:]
            e_values = e.values.numpy()[idx]
            e_new = tf.sparse.SparseTensor(indices=e_indices, values=e_values, dense_shape=[n_points, n_points])
            e_new = tf.sparse.reorder(e_new)
            adj = tf.sparse.to_dense(e_new)
            return(adj)
        # plot affinity matrix against adjacency
        affinity = self.get_affinity_matrix(X, sigmoid_flag=True)
        dataset_name = self.dataset_name
        adj = make_dense(edge_index, n_points)
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
            savefile=f"{results_folder}aff_plots/{dataset_name}.png"
            plt.savefig(savefile)
            plt.close()
        else:   
            plt.show()
