import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import DataLoader
from torch_geometric.datasets import DBLP
from torch_geometric.nn import HANConv,GCNConv,to_hetero,SAGEConv ,GraphConv ,GCNConv
import argparse
from get_dataset import set_random_seeds,get_dataset,calculate_label_accuracy ,calculate_f1_score
import torch.nn.functional as F
import torch_geometric.transforms as T
from tqdm import tqdm
from sklearn.metrics import f1_score
from sklearn.metrics import normalized_mutual_info_score
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score

parser = argparse.ArgumentParser(description='hetedis')
parser.add_argument('--seed', type=int, default=80)
parser.add_argument('--dataset', type=str, default='DBLP')
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--gamma', type=float, default=1.0)
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--model', type=str, default='K-GNNs')
args = parser.parse_args()

set_random_seeds(args.seed)
# 构建模型
class KGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(KGNN, self).__init__()
        # H, D = self.heads, self.out_channels // self.heads
        self.conv1 = GraphConv(in_channels, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, out_channels)
        self.lin = nn.Linear(out_channels, num_classes)

    def forward( self, x, edge_index):
        
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x=self.lin(x)

        return x

def train():
    model = KGNN(-1, 128, num_classes)
    model = to_hetero(model, graph.metadata()).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=2e-4)
    loss_function = torch.nn.CrossEntropyLoss().to(device)
    min_epochs = 5
    
    best_val_acc = 0
    final_best_acc = 0
    final_F1 = 0
    final_F1_bias = 0
    final_F1_mean = 0
    for epoch in tqdm(range(150)):
        
        model.train()
        out = model(graph.x_dict, graph.edge_index_dict)
        loss = loss_function(out['author'][train_mask], graph['author'].y[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        val_acc, val_loss,va,_, __ ,_,___= test(model, val_mask)
        test_acc, test_loss, label_accuracy, label_f1_scores, F1_test,F1_mean, F1_bias = test(model, test_mask)
        
        if epoch + 1 > min_epochs and val_acc > best_val_acc:
            best_val_acc = val_acc
            final_best_acc = test_acc
            final_F1 = F1_test
            final_F1_bias = F1_bias
            final_F1_mean = F1_mean
            torch.save({'model': model.state_dict()}, f'./model_save_param/{args.model}/{args.dataset}/{args.dataset}_model_seed_{args.seed}.pt')  
        
        tqdm.write('Epoch{:3d} Train Loss {:.5f}  Val Acc {:.3f} Test Acc {:.3f}'.format(epoch, loss.item(),
                                                                                                        val_acc, test_acc))   
        
         
        for label, f1 in label_f1_scores.items():
            tqdm.write('Label: {} F1 Score: {:.3f}'.format(label, f1))
    
    return final_best_acc,final_F1,final_F1_bias,final_F1_mean


def test(model, mask):
    model.eval()
    with torch.no_grad():
        out = model(graph.x_dict , graph.edge_index_dict)
        
    _, pred = out['author'].max(dim=1)

    correct = int(pred[mask].eq(graph['author'].y[mask]).sum().item())
    acc = correct / int(mask.sum())
    label_accuracy = calculate_label_accuracy(pred, y, mask)
    loss_function = torch.nn.CrossEntropyLoss().to(device)
    loss = loss_function(out['author'][mask], graph['author'].y[mask])
    
    label_f1_scores ,F1_test ,F1_bias ,F1_mean= calculate_f1_score(pred, y, mask)
    
        
    return acc, loss.item(), label_accuracy ,label_f1_scores ,F1_test ,F1_bias,F1_mean


device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

graph, num_classes, dataset = get_dataset('./data',args.dataset ,transform=T.NormalizeFeatures())  
graph = graph.to(device)
train_mask, val_mask, test_mask = graph['author'].train_mask, graph['author'].val_mask, graph['author'].test_mask
y = graph['author'].y
'''
graph, num_classes, dataset = get_dataset('./data/IMDB',args.dataset ,transform=T.NormalizeFeatures())  
graph = graph.to(device)
train_mask, val_mask, test_mask = graph['movie'].train_mask, graph['movie'].val_mask, graph['movie'].test_mask
y = graph['movie'].y


graph, num_classes = get_dataset('./data',args.dataset ,transform=T.NormalizeFeatures())  
graph = graph.to(device)
train_mask, val_mask, test_mask = graph['paper'].train_mask, graph['paper'].val_mask, graph['paper'].test_mask
y = graph['paper'].y
'''
final_best_acc,final_F1,final_F1_bias,final_F1_mean = train()

print('GraphConv F1-sorce:', "%0.2f%%" % (final_F1 * 100))
print('GraphConv F1-mean:', "%0.2f%%" % (final_F1_mean * 100))
print('GraphConv F1-bias:', "%0.2f%%" % (final_F1_bias * 100))