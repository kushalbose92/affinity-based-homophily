"""
program: utility routines for affinity matrix computation script
author: indranil ojha

"""

# import libraries
from torch_geometric.datasets import Planetoid, WebKB, Actor, WikipediaNetwork
from torch_geometric.datasets import KarateClub, FacebookPagePage, EmailEUCore
from torch_geometric.datasets import TUDataset, Airports, Amazon, Reddit, Flickr
from torch_geometric.datasets import Yelp, GitHub, HeterophilousGraphDataset
from torch_geometric.datasets import FakeDataset, BAShapes
from torch_geometric.transforms import NormalizeFeatures, RemoveIsolatedNodes, Compose, AddSelfLoops, LargestConnectedComponents
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

def normalize_adj(edge_index, n_nodes):
    def d_half_inv(e, n_nodes):
        deg = [e.count(i) for i in range(n_nodes)]
        deg_inv_sqrt = [1/np.sqrt(d) if d!=0 else 0 for d in deg]
        return(deg_inv_sqrt)
    edges = edge_index.indices.numpy()
    vals = list(edge_index.values.numpy())
    n_edges = len(edges)
    deg_inv_sqrt_0 = d_half_inv(list(edges[:,0]), n_nodes)
    deg_inv_sqrt_1 = d_half_inv(list(edges[:,1]), n_nodes)
    v_norm = [vals[i]*deg_inv_sqrt_0[edges[i,0]]*deg_inv_sqrt_1[edges[i,1]] 
              for i in range(n_edges)]
    edge_index = tf.sparse.SparseTensor(indices=edge_index.indices, 
                    values=v_norm, dense_shape=[n_nodes, n_nodes])
    return(edge_index)

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

def load_snap_patents(data_folder):
    fname = f"{data_folder}snap_patents/snap_patents.mat"
    patents_dict = loadmat(fname)
    X = torch.from_numpy(csc_matrix.toarray(patents_dict['node_feat']))
    y = torch.from_numpy(patents_dict['years'][0])
    edge_index = torch.from_numpy(patents_dict['edge_index'])
    n_classes = len(np.unique(y))

    data = Data(x=X, y=y, edge_index=edge_index, is_directed=True)
    return(data, n_classes)

def load_arxiv(folder):
    data_folder = f"{folder}arxiv-year/ogbn_arxiv/raw/"
    mask_fname = f"{folder}arxiv-year/splits/arxiv-year-splits.npy"

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
def load_dataset(data_folder, dataset_name, frac=1, rand_train=False, 
              remove_isolated=False, add_self_loops=False, largest_connected_components=False):
        
    if dataset_name == 'SnapPatents':
        data, n_classes = load_snap_patents(data_folder)
    elif dataset_name == 'ogbn-arxiv':
        data, n_classes = load_arxiv(data_folder)
    else:
        if dataset_name in ['Cora', 'CiteSeer', 'PubMed']:
            dataset = Planetoid(root=f'{data_folder}pyG', name=dataset_name)
        if dataset_name in ['Cornell', 'Wisconsin', 'Texas']:
            dataset = WebKB(root=f'{data_folder}pyG', name=dataset_name)
        if dataset_name in ['Squirrel', 'Chameleon']:
            dataset = WikipediaNetwork(root=f'{data_folder}pyG', name=dataset_name)
        if dataset_name in ['KarateClub']:
            dataset = KarateClub()
        if dataset_name in ['Actor']:
            dataset = Actor(root=f'{data_folder}pyG/Actor')
        if dataset_name in ['FacebookPagePage']:
            dataset = FacebookPagePage(root=f'{data_folder}pyG/{dataset_name}')
        if dataset_name in ['USA', 'Europe', 'Brazil']:
            dataset = Airports(root=f'{data_folder}pyG/{dataset_name}')
        if dataset_name in ['EmailEUCore']:
            dataset = EmailEUCore(root=f'{data_folder}pyG/{dataset_name}')
        if dataset_name in ["IMDB-BINARY", "REDDIT-BINARY", "PROTEINS"]:
            dataset = TUDataset(root=f'{data_folder}pyG', name=dataset_name)
        if dataset_name in ["Computers", "Photo"]:
            dataset = Amazon(root=f'{data_folder}pyG', name=dataset_name)
        if dataset_name in ["Reddit"]:
            dataset = Reddit(root=f'{data_folder}pyG/{dataset_name}')
        if dataset_name in ["Flickr"]:
            dataset = Flickr(root=f'{data_folder}pyG/{dataset_name}')
        if dataset_name in ["Yelp"]:
            dataset = Yelp(root=f'{data_folder}pyG/{dataset_name}')
        if dataset_name in ["GitHub"]:
            dataset = GitHub(root=f'{data_folder}pyG/{dataset_name}')
        if dataset_name in ["Roman-empire", "Amazon-ratings", "Minesweeper", "Tolokers", "Questions"]:
            dataset = HeterophilousGraphDataset(root=f'{data_folder}pyG/{dataset_name}', 
                        name = dataset_name)
        if dataset_name in ['FakeDataset']:
            dataset = FakeDataset()
        if dataset_name in ['BAShapes']:
            dataset =   BAShapes()
    
        n_classes = dataset.num_classes 
        data = dataset[0]  # Get the first graph object.

    # transform as per requirement
    transform=[NormalizeFeatures()]
    if remove_isolated:
        transform.append(RemoveIsolatedNodes())
    if add_self_loops:
        transform.append(AddSelfLoops())
    if largest_connected_components:
        transform.append(LargestConnectedComponents())
    transform = Compose(transform)
    data = transform(data)

    if frac<1: # subgraph wanted
        data = get_subgraph(data, frac)

    return(data, n_classes)

def load_data(data_folder, dataset_name, frac=1, rand_train=False, 
              remove_isolated=False, add_self_loops=False, largest_connected_components=False):
    try:
        data, n_classes = load_dataset(data_folder, dataset_name, frac=1, rand_train=False, 
                      remove_isolated=False, add_self_loops=False, largest_connected_components=False)
        X, y, edge_index = data.x, data.y, data.edge_index
        is_directed = data.is_directed()
        n_nodes, n_features = X.shape
        n_edges = edge_index.shape[1]
        #X, y = torch_to_tf(X), torch_to_tf(y)
        X, y = tf.convert_to_tensor(X.numpy()), tf.convert_to_tensor(y.numpy())

        edge_index = tf.cast(edge_index, tf.int64)
        edge_index = tf.sparse.SparseTensor(indices=tf.transpose(edge_index), values=tf.ones(n_edges), dense_shape=[n_nodes, n_nodes])
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
        X, y, edge_index, train_mask, test_mask, is_directed = None, None, None, None, None, None
        error_status = 1

    return(X, y, edge_index, train_mask, test_mask, n_classes, is_directed, error_status)

