import random
import os
import argparse
from sklearn.metrics import f1_score
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch_geometric import datasets
from torch_geometric.loader import NeighborLoader
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.datasets import DBLP
import torch_geometric.transforms as T
from torch_geometric.utils import degree
import torch.nn.functional as F
from load_acm import get_binary_mask,load_acm_raw
import torch.nn.functional as F

parser = argparse.ArgumentParser(description='dataset')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--dataset', type=str, default='IMDB')
parser.add_argument('--device', type=int, default=0)
args = parser.parse_args()
device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
def set_random_seeds(random_seed):
    r"""Sets the seed for generating random numbers."""
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(random_seed)
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)

def get_dataset(root, dataname, transform=NormalizeFeatures(), train_mask_ratio=0.6, val_mask_ratio=0.2):
    pyg_dataset_dict = {
        'DBLP': (datasets.DBLP, 'DBLP_processed'),
        'AMiner': (datasets.AMiner, 'AMiner'),
        'IMDB': (datasets.IMDB, 'IMDB')
    }
    
    if dataname == 'ACM':
        graph , num_nodes= load_acm_raw(remove_self_loop = False)
        num_classes = torch.max(graph['paper'].y).item() + 1
        
        num_nodes = graph['paper'].num_nodes
        num_train = int(train_mask_ratio * num_nodes)
        num_val = int(val_mask_ratio * num_nodes)
        num_test = num_nodes - num_train - num_val

        indices = torch.arange(num_nodes)
        shuffled_indices = torch.randperm(num_nodes)

        train_indices = shuffled_indices[:num_train]
        val_indices = shuffled_indices[num_train:num_train+num_val]
        test_indices = shuffled_indices[num_train+num_val:]

        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        train_mask[train_indices] = True
        val_mask[val_indices] = True
        test_mask[test_indices] = True

        graph['paper'].train_mask = train_mask
        graph['paper'].val_mask = val_mask
        graph['paper'].test_mask = test_mask
        
        train_label_counts = torch.bincount(graph['paper'].y[train_mask])
        val_label_counts = torch.bincount(graph['paper'].y[val_mask])
        test_label_counts = torch.bincount(graph['paper'].y[test_mask])
        for label in range(num_classes):
            print("Label {}: Train Nodes: {}, Val Nodes: {}, Test Nodes: {}".format(label, train_label_counts[label], val_label_counts[label], test_label_counts[label]))
            label_counts = torch.bincount(graph['paper'].y)
        return graph ,num_classes   
        

    dataset_class, name = pyg_dataset_dict[dataname]
    if name =='DBLP_processed' :
    
        dataset = dataset_class(root, transform=transform)
        graph = dataset[0]
        graph['conference'].x = torch.ones((graph['conference'].num_nodes, 1))
        num_classes = torch.max(graph['author'].y).item() + 1

        
        num_nodes = graph['author'].num_nodes
        num_train = int(train_mask_ratio * num_nodes)
        num_val = int(val_mask_ratio * num_nodes)
        num_test = num_nodes - num_train - num_val

        indices = torch.arange(num_nodes)
        shuffled_indices = torch.randperm(num_nodes)

        train_indices = shuffled_indices[:num_train]
        val_indices = shuffled_indices[num_train:num_train+num_val]
        test_indices = shuffled_indices[num_train+num_val:]

        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        train_mask[train_indices] = True
        val_mask[val_indices] = True
        test_mask[test_indices] = True

        graph['author'].train_mask = train_mask
        graph['author'].val_mask = val_mask
        graph['author'].test_mask = test_mask
        
        print(graph)
        train_label_counts = torch.bincount(graph['author'].y[train_mask])
        val_label_counts = torch.bincount(graph['author'].y[val_mask])
        test_label_counts = torch.bincount(graph['author'].y[test_mask])

        for label in range(num_classes):
            print("Label {}: Train Nodes: {}, Val Nodes: {}, Test Nodes: {}".format(label, train_label_counts[label], val_label_counts[label], test_label_counts[label]))
            label_counts = torch.bincount(graph['author'].y)
            
    if name =='IMDB':
        dataset = dataset_class(root, transform=transform)
        graph = dataset[0]
        print(graph)
        num_classes = torch.max(graph['movie'].y).item() + 1
        print(num_classes)
        
        num_nodes = graph['movie'].num_nodes
        num_train = int(train_mask_ratio * num_nodes)
        num_val = int(val_mask_ratio * num_nodes)
        num_test = num_nodes - num_train - num_val

        indices = torch.arange(num_nodes)
        shuffled_indices = torch.randperm(num_nodes)

        train_indices = shuffled_indices[:num_train]
        val_indices = shuffled_indices[num_train:num_train+num_val]
        test_indices = shuffled_indices[num_train+num_val:]

        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        train_mask[train_indices] = True
        val_mask[val_indices] = True
        test_mask[test_indices] = True

        graph['movie'].train_mask = train_mask
        graph['movie'].val_mask = val_mask
        graph['movie'].test_mask = test_mask


        train_label_counts = torch.bincount(graph['movie'].y[train_mask])
        val_label_counts = torch.bincount(graph['movie'].y[val_mask])
        test_label_counts = torch.bincount(graph['movie'].y[test_mask])

        for label in range(num_classes):
            print("Label {}: Train Nodes: {}, Val Nodes: {}, Test Nodes: {}".format(label, train_label_counts[label], val_label_counts[label], test_label_counts[label]))
            label_counts = torch.bincount(graph['movie'].y)
            
            
        

    return graph, num_classes, dataset
  
        
def calculate_label_accuracy(pred, y, mask):
    label_accuracy = {}
    unique_labels = torch.unique(y)
    for label in unique_labels:
        label_mask = (y == label)
        label_correct = int(pred[mask & label_mask].eq(y[mask & label_mask]).sum().item())
        label_accuracy[label.item()] = label_correct / int((mask & label_mask).sum())
    return label_accuracy

def calculate_f1_score(pred, y, mask):
    pred = pred[mask].cpu().numpy()
    y = y[mask].cpu().numpy()
    f1_scores = {}
    
    F1_test = f1_score(y, pred, average='micro')
    
    unique_labels = set(y)
    for label in unique_labels:
        label_pred = pred[y == label]
        label_true = y[y == label]
        f1 = f1_score(label_true, label_pred, average='micro', labels=[label])
        f1_scores[label] = f1
    
    # 计算最大值和最小值
    max_f1 = max(f1_scores.values())
    min_f1 = min(f1_scores.values())

    # 计算差异
    f1_difference = max_f1 - min_f1
    
    # 计算每个标签的F1分数的均值
    mean_f1 = np.mean(list(f1_scores.values()))
    
    # 计算每个标签的F1分数与均值方差
    #variance_f1 = np.var([(score - mean_f1) ** 2 for score in f1_scores.values()])
    
    return f1_scores , F1_test ,mean_f1, f1_difference

def compute_label_loss(output, target, num_classes):
    loss_function = torch.nn.CrossEntropyLoss().to(device)
    label_loss = []
    for label_idx in range(num_classes):
        label_mask = (target == label_idx)
        label_output = output[label_mask]
        label_target = target[label_mask]
        label_loss.append(loss_function(label_output, label_target))
    
    loss_tensor = torch.stack(label_loss)
    max_loss = loss_tensor.max()
    min_loss = loss_tensor.min()
    print(max_loss)
    
    return max_loss ,min_loss

def focal_loss(inputs, targets, alpha=0.1, gamma=2.3):
    #GraphSage:DBLP,ACM,IMDB:0.1 1.5
    ce_loss = F.cross_entropy(inputs, targets, reduction='none')  
    pt = torch.exp(-ce_loss)  
    focal_loss = alpha * (1 - pt) ** gamma * ce_loss


    return torch.mean(focal_loss)
