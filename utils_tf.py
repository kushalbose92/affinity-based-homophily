"""
program: utility routines on GCN

"""
# import libraries
from torch_geometric.datasets import Planetoid, WebKB, Actor, WikipediaNetwork
from torch_geometric.datasets import KarateClub, FacebookPagePage, EmailEUCore
from torch_geometric.datasets import TUDataset, Airports, Amazon, Reddit, Flickr
from torch_geometric.datasets import Yelp, GitHub, HeterophilousGraphDataset
from torch_geometric.datasets import FakeDataset, BAShapes
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import to_torch_coo_tensor
from torch_geometric.data import Data
from scipy.io import loadmat
from scipy.sparse import csc_matrix
import tensorflow as tf
import torch
import numpy as np

# convert torch tensor to tf tensor
def torch_to_tf(torch_tensor):
    np_tensor = torch_tensor.numpy()
    tf_tensor = tf.convert_to_tensor(np_tensor)
    return(tf_tensor)

# create trtain and test masks where not present
def get_mask(n_nodes):
    mask = np.array([1]*n_nodes, dtype=bool)
    step =  10 # 10% for test, uniform & not random to ensure reproducability 
    for i in range(round(n_nodes/step)):
        mask[step*i] = False
    train_mask = tf.reshape(tf.convert_to_tensor(mask), (-1,1))
    test_mask = tf.math.logical_not(train_mask)
    return(train_mask, test_mask)

def normalize_adj(adj):
    deg_sqrt = tf.sqrt(tf.reduce_sum(adj, axis=1))
    deg_sqrt = tf.linalg.diag(deg_sqrt)
    deg_inv_sqrt = tf.linalg.pinv(deg_sqrt)
    norm_adj = tf.linalg.matmul(deg_inv_sqrt, tf.linalg.matmul(adj, deg_inv_sqrt))
    return(norm_adj)

def get_subgraph(graph, frac):
    n_nodes = len(graph.y)
    n_selected_nodes = int(frac*n_nodes)
    bool_t = [True] * n_selected_nodes
    bool_f = [False] * (n_nodes - n_selected_nodes)
    bool_tf = torch.tensor(bool_t + bool_f, dtype=torch.bool)
    perm = torch.randperm(n_nodes)
    idx = bool_tf[perm]
    subgraph = graph.subgraph(idx)
    return(subgraph)

# remove isolated nodes
def remove_isolated_nodes(graph):
    adj = to_torch_coo_tensor(graph.edge_index).to_dense()
    n_nodes = len(graph.y)
    deg = torch.sum(adj, dim=1)
    isolated_nodes = np.where(deg==0)[0]
    mask = torch.from_numpy(np.array([True for _ in range(n_nodes)]))
    mask[isolated_nodes] = False
    subgraph = graph.subgraph(mask)
    return(subgraph)

def load_snap_patents():
    fname = "D:/Indranil/ML2/datasets/snap_patents/snap_patents.mat"
    patents_dict = loadmat(fname)
    X = torch.from_numpy(csc_matrix.toarray(patents_dict['node_feat']))
    y = torch.from_numpy(patents_dict['years'][0])
    edge_index = torch.from_numpy(patents_dict['edge_index'])
    n_classes = len(np.unique(y))

    data = Data(x=X, y=y, edge_index=edge_index, is_directed=True)
    return(data, n_classes)

def load_arxiv_year():
    data_folder = "D:/Indranil/ML2/datasets/arxiv-year/ogbn_arxiv/raw/"
    mask_fname = "D:/Indranil/ML2/datasets/arxiv-year/splits/arxiv-year-splits.npy"

    X = np.loadtxt(f'{data_folder}node-feat.csv', delimiter=',')
    y = np.loadtxt(f'{data_folder}node_year.csv', delimiter=',')
    edge_index = np.loadtxt(f'{data_folder}edge.csv', delimiter=',', dtype=int)

    X = torch.from_numpy(X)
    y = torch.from_numpy(y)
    edge_index = torch.from_numpy(np.transpose(edge_index))
    n_classes = len(np.unique(y))
    n_nodes = len(y)
    
    mask_dict = np.load(mask_fname, allow_pickle=True)
    n_split = len(mask_dict)
    train_mask = []
    test_mask = []
    for i in range(n_split):
        train_mask_idx = mask_dict[i]['train']
        train_mask_bool = np.zeros(n_nodes, dtype=bool)
        train_mask_bool[train_mask_idx] = True
        train_mask.append(train_mask_bool)
        test_mask_idx = mask_dict[i]['test']
        test_mask_bool = np.zeros(n_nodes, dtype=bool)
        test_mask_bool[test_mask_idx] = True
        test_mask.append(test_mask_bool)
    train_mask = np.transpose(np.array(train_mask))
    test_mask = np.transpose(np.array(test_mask))
    train_mask = torch.from_numpy(train_mask)
    test_mask = torch.from_numpy(test_mask)

    data = Data(x=X, y=y, edge_index=edge_index, 
            train_mask=train_mask, test_mask=test_mask, is_directed=True)
    return(data, n_classes)

# load dataset
def load_data(dataset_name, frac=1, rand_train=False, remove_isolated=False):
    if dataset_name == 'SnapPatents':
        data, n_classes = load_snap_patents()
    elif dataset_name == 'ArxivYear':
        data, n_classes = load_arxiv_year()
    else:
        if dataset_name in ['Cora', 'CiteSeer', 'PubMed']:
            dataset = Planetoid(root='D:/Indranil/ML2/datasets/pyG', name=dataset_name, transform=NormalizeFeatures())
        if dataset_name in ['cornell', 'wisconsin', 'texas']:
            dataset = WebKB(root='D:/Indranil/ML2/datasets/pyG', name=dataset_name, transform=NormalizeFeatures())
        if dataset_name in ['squirrel', 'chameleon']:
            dataset = WikipediaNetwork(root='D:/Indranil/ML2/datasets/pyG', name=dataset_name, transform=NormalizeFeatures())
        if dataset_name in ['KarateClub']:
            dataset = KarateClub(transform=NormalizeFeatures())
        if dataset_name in ['Actor']:
            dataset = Actor(root='D:/Indranil/ML2/datasets/pyG/Actor', transform=NormalizeFeatures())
        if dataset_name in ['FacebookPagePage']:
            dataset = FacebookPagePage(root=f'D:/Indranil/ML2/datasets/pyG/{dataset_name}', transform=NormalizeFeatures())
        if dataset_name in ['USA', 'Europe', 'Brazil']:
            dataset = Airports(root=f'D:/Indranil/ML2/datasets/pyG/{dataset_name}', transform=NormalizeFeatures())
        if dataset_name in ['EmailEUCore']:
            dataset = EmailEUCore(root=f'D:/Indranil/ML2/datasets/pyG/{dataset_name}', transform=NormalizeFeatures())
        if dataset_name in ["IMDB-BINARY", "REDDIT-BINARY", "PROTEINS"]:
            dataset = TUDataset(root='D:/Indranil/ML2/datasets/pyG', name=dataset_name, transform=NormalizeFeatures())
        if dataset_name in ["Computers", "Photo"]:
            dataset = Amazon(root='D:/Indranil/ML2/datasets/pyG', name=dataset_name, transform=NormalizeFeatures())
        if dataset_name in ["Reddit"]:
            dataset = Reddit(root=f'D:/Indranil/ML2/datasets/pyG/{dataset_name}', transform=NormalizeFeatures())
        if dataset_name in ["Flickr"]:
            dataset = Flickr(root=f'D:/Indranil/ML2/datasets/pyG/{dataset_name}', transform=NormalizeFeatures())
        if dataset_name in ["Yelp"]:
            dataset = Yelp(root=f'D:/Indranil/ML2/datasets/pyG/{dataset_name}', transform=NormalizeFeatures())
        if dataset_name in ["GitHub"]:
            dataset = GitHub(root=f'D:/Indranil/ML2/datasets/pyG/{dataset_name}', transform=NormalizeFeatures())
        if dataset_name in ["Roman-empire", "Amazon-ratings", "Minesweeper", "Tolokers", "Questions"]:
            dataset = HeterophilousGraphDataset(root=f'D:/Indranil/ML2/datasets/pyG/{dataset_name}', 
                        name = dataset_name, transform=NormalizeFeatures())
        if dataset_name in ['FakeDataset']:
            dataset = FakeDataset(transform=NormalizeFeatures())
        if dataset_name in ['BAShapes']:
            dataset =   BAShapes(transform=NormalizeFeatures())
    
        n_classes = dataset.num_classes 
        data = dataset[0]  # Get the first graph object.

    if remove_isolated:
        data = remove_isolated_nodes(data)

    if frac<1: # subgraph wanted
        data = get_subgraph(data, frac)
    try:
        X, y, edge_index = data.x, data.y, data.edge_index
        is_directed = data.is_directed()
        n_nodes, n_features = X.shape
        #X, y = torch_to_tf(X), torch_to_tf(y)
        X, y = tf.convert_to_tensor(X.numpy()), tf.convert_to_tensor(y.numpy())

        adj_sparse = tf.cast(edge_index, tf.int64)
        adj = tf.cast(to_torch_coo_tensor(edge_index).to_dense(), tf.float32)
        if hasattr(data,'train_mask') and rand_train == False:
            train_mask, test_mask = data.train_mask.numpy(), data.test_mask.numpy()
            if np.ndim(train_mask)==1:
                train_mask = train_mask.reshape(-1,1)
                test_mask = test_mask.reshape(-1,1)
            train_mask = tf.convert_to_tensor(train_mask)
            test_mask = tf.convert_to_tensor(test_mask)
        else:
            train_mask, test_mask = get_mask(n_nodes)
        error_status = 0
    except:
        X, y, adj_sparse, adj, train_mask, test_mask, is_directed = None, None, None, None, None, None, None
        error_status = 1

    return(X, y, adj_sparse, adj, train_mask, test_mask, n_classes, is_directed, error_status)

# compute edge-homophily of dataset
def compute_homophily(y, adj_sparse):
    u_list = adj_sparse[0,:].numpy()
    v_list = adj_sparse[1,:].numpy()
    y_np = y.numpy()
    y_u = np.array([y_np[u] for u in u_list])
    y_v = np.array([y_np[v] for v in v_list])
    like_edges = sum(y_u==y_v)
    total_edges = adj_sparse.shape[1]
    homophily = like_edges/total_edges
    return(homophily.item())

def rbo(list1, list2, p=0.5):
   # tail recursive helper function
   def helper(ret, i, d):
       l1 = set(list1[:i]) if i < len(list1) else set(list1)
       l2 = set(list2[:i]) if i < len(list2) else set(list2)
       a_d = len(l1.intersection(l2))/i
       term = pow(p, i) * a_d
       if d == i:
           return ret + term
       return helper(ret + term, i + 1, d)
   k = max(len(list1), len(list2))
   x_k = len(set(list1).intersection(set(list2)))
   summation = helper(0, 1, k)
   return ((float(x_k)/k) * pow(p, k)) + ((1-p)/p * summation)

def add_self_loop(adj, adj_sparse):
    n_nodes = adj.shape[0]
    self_loop_edges = np.transpose(np.array([[i,i] for i in range(n_nodes)]))
    adj_sparse = tf.concat([adj_sparse, self_loop_edges], axis=1)
    adj = adj + np.eye(n_nodes)
    return(adj, adj_sparse)

# create 2D mask from 1D to fit the adj matrix
def get_2D_mask(mask):
    mask = tf.cast(mask, tf.float32)
    mask = tf.linalg.matmul(mask, tf.transpose(mask))
    return(mask)

# given true and predicted values of binary tensor
# compute TP, FP, TN, FN & F1-score
def compute_F1(y_true, y_pred):
    y_diff = y_true - y_pred
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

# loading the diagonal affinity matrix i.e. only the normalized self-affinity values)
def load_aff_matrix(filename, n_edges):
    aff_matrix = np.loadtxt(filename, delimiter=',') # load 2D matrix
    aff_matrix = tf.cast(aff_matrix, tf.float32)
    n_nodes = aff_matrix.shape[0]
    all_sum = tf.reduce_sum(aff_matrix).numpy()
    diag_sum = tf.reduce_sum(tf.linalg.diag_part(aff_matrix)).numpy()
    non_diag_sum = all_sum - diag_sum
    #diag_mean = diag_sum/self.n_nodes
    non_diag_mean = non_diag_sum/(n_nodes * (n_nodes - 1))
    aff_matrix_diag = np.diagonal(aff_matrix) # 1D matrix containing diag elements
    normalized_diag = np.log(aff_matrix_diag/non_diag_mean) /np.log((n_nodes * (n_nodes - 1))/n_edges)
    normalized_diag = np.clip(normalized_diag,0, np.max(normalized_diag))
    diagonilized_normalized_aff_matrix = tf.linalg.diag(normalized_diag) # 2D diagonal matrix given the elements
    diagonilized_normalized_aff_matrix = diagonilized_normalized_aff_matrix*2.0 - 1
    return(diagonilized_normalized_aff_matrix)
