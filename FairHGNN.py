import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import DataLoader
from torch_geometric.datasets import DBLP
from torch_geometric.nn import HANConv ,SAGEConv ,to_hetero
from torch_geometric.utils import to_undirected
import argparse
from get_dataset import set_random_seeds,get_dataset,calculate_label_accuracy ,calculate_f1_score ,poly_focal_loss,get_homo_dataset
import torch.nn.functional as F
import torch_geometric.transforms as T
from tqdm import tqdm
from sklearn.metrics import f1_score
from sklearn.metrics import normalized_mutual_info_score
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score

parser = argparse.ArgumentParser(description='dblp')

parser.add_argument('--seed', type=int, default=2)
parser.add_argument('--dataset', type=str, default='ACM')

parser.add_argument('--beta', type=float, default=0.8)
parser.add_argument('--alhpa', type=float, default=0.1)
parser.add_argument('--x_class', type=str, default='paper')
parser.add_argument('--T', type=float, default=0.25)
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--model', type=str, default='FairHGNN')

args = parser.parse_args()

set_random_seeds(args.seed)

# model
class GraphSage(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSage, self).__init__()
        # H, D = self.heads, self.out_channels // self.heads
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.relu = nn.ReLU()
        self.Lin = nn.Linear(out_channels, num_classes)
        
    def forward( self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        
        x=self.Lin(x)

        return x

def train(epoch):
    model.train()
    optimizer.zero_grad()
    
        
    teacher_output = model(graph.x_dict, Tea_edge_index_dict) 
    teacher_output = teacher_output[args.x_class]/args.T


    out = model(graph.x_dict, graph.edge_index_dict)
    out = out[args.x_class]


    _, pred = (out.max(dim=1))      

    focalloss = poly_focal_loss(alpha, out[train_mask], graph[args.x_class].y[train_mask])

    label_f1_scores ,F1_test ,F1_bias ,F1_mean= calculate_f1_score(pred, y, train_mask)     

    loss = args.beta * loss_function(out[train_mask], graph[args.x_class].y[train_mask])
    if epoch > 20 : #20#
        
        loss += args.alhpa * KLloss(F.log_softmax(teacher_output, dim=-1),
                                      F.log_softmax(out, dim=-1))
    
    
    loss += (1-args.beta) * focalloss
   
        
    loss.backward()
    optimizer.step()
    return loss


def test(model, mask):
    model.eval()
    with torch.no_grad():
        out = model(graph.x_dict, graph.edge_index_dict)
    out = out[args.x_class]
    
    _, pred = out.max(dim=1)

    correct = int(pred[mask].eq(y[mask]).sum().item())
    acc = correct / int(mask.sum())
    label_accuracy = calculate_label_accuracy(pred, y, mask)
    loss_function = torch.nn.CrossEntropyLoss().to(device)
    loss = loss_function(out[mask], y[mask])
    
    label_f1_scores ,F1_test ,F1_bias ,F1_mean= calculate_f1_score(pred, y, mask)
    
        
    return acc, loss.item(), label_accuracy ,label_f1_scores ,F1_test ,F1_bias,F1_mean


device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
'''
graph, num_classes, dataset = get_dataset('./data',args.dataset ,transform=T.NormalizeFeatures())  
graph = graph.to(device)
train_mask, val_mask, test_mask = graph['author'].train_mask, graph['author'].val_mask, graph['author'].test_mask
y = graph['author'].y
Tea_edge_index = torch.tensor([[], []], dtype=torch.int64, device=device)
empty_edge_index = torch.empty((2, 0), dtype=torch.long, device='cuda:0')
Tea_edge_index_dict = {
    ('author','to', 'paper') : empty_edge_index,
    ('paper', 'to', 'author') : empty_edge_index,
    ('paper', 'to', 'term') : empty_edge_index,
    ('paper', 'to', 'conference') : empty_edge_index,
    ('term', 'to', 'paper') : empty_edge_index,
    ('conference', 'to', 'paper') : empty_edge_index,
    ('author', 'to', 'author') : empty_edge_index
}
alpha = torch.tensor([1/(733+224+240), 1/(745+745), 1000/(655+240+214), 1000/(589+209+208)],)
alpha = alpha.to(device)

graph, num_classes, dataset = get_dataset('./data/IMDB',args.dataset ,transform=T.NormalizeFeatures())  
graph = graph.to(device)
train_mask, val_mask, test_mask = graph['movie'].train_mask, graph['movie'].val_mask, graph['movie'].test_mask
y = graph['movie'].y
Tea_edge_index = torch.tensor([[], []], dtype=torch.int64, device=device)
empty_edge_index = torch.empty((2, 0), dtype=torch.long, device='cuda:0')
Tea_edge_index_dict = {
    ('movie', 'to', 'director'): empty_edge_index,
    ('movie', 'to', 'actor'): empty_edge_index,
    ('director', 'to', 'movie'): empty_edge_index,
    ('actor', 'to', 'movie'): empty_edge_index,
    ('movie', 'to', 'movie'): empty_edge_index
}

alpha = torch.tensor([1/(1134+231), 1000/(937+338+309), 1000/(947+292+320)],)
alpha = alpha.to(device)

'''
graph, num_classes = get_dataset('./data',args.dataset ,transform=T.NormalizeFeatures())  
graph = graph.to(device)
train_mask, val_mask, test_mask = graph['paper'].train_mask, graph['paper'].val_mask, graph['paper'].test_mask
y = graph['paper'].y
Tea_edge_index = torch.tensor([[], []], dtype=torch.int64, device=device)
empty_edge_index = torch.empty((2, 0), dtype=torch.long, device='cuda:0')
Tea_edge_index_dict = {
    ('author', 'ap', 'paper'): empty_edge_index,
    ('field', 'fp', 'paper'): empty_edge_index,
    ('paper', 'pa', 'author'): empty_edge_index,
    ('paper', 'pf', 'field'): empty_edge_index,
    ('paper', 'to', 'paper'): empty_edge_index
}
alpha = torch.tensor([100/(753+273+247), 100/(1198+393+403), 100/(591+181+198)],)
alpha = alpha.to(device)


model = GraphSage(-1, 256, num_classes).to(device)
model = to_hetero(model, graph.metadata()).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
loss_function = torch.nn.CrossEntropyLoss().to(device)
KLloss = torch.nn.KLDivLoss(reduction="batchmean", log_target=True) 


min_epochs = 5
best_val_acc = 0
final_best_acc = 0
final_F1 = 0
final_F1_bias = 0
final_F1_mean = 0
patience = 0
for epoch in tqdm(range(400)):
    
    

    model.train()
    loss= train(epoch)    

    val_acc, val_loss,va,_, __ ,val_F1_mean,val_F1_bias= test(model, val_mask )
    test_acc, test_loss, label_accuracy, label_f1_scores, F1_test,F1_mean, F1_bias = test(model, test_mask)
    patience += 1
    if epoch + 1 > min_epochs and val_acc > best_val_acc:
        patience = 0
        best_val_acc = val_acc
        final_best_acc = test_acc
        final_F1 = F1_test
        final_F1_bias = F1_bias
        final_F1_mean = F1_mean
        torch.save({'model': model.state_dict()}, f'./model_save_param/{args.model}/{args.dataset}/{args.dataset}_distill_HeBias_model_seed_{args.seed}.pt')  
        for label, f1 in label_f1_scores.items():
            tqdm.write('Label: {} F1 Score: {:.3f}'.format(label, f1))

        tqdm.write('Epoch{:3d} Train Loss {:.5f}  Val Acc {:.3f} Test Acc {:.3f}'.format(epoch, loss.item(),
                                                                                                        val_acc, test_acc))
    if patience > 100:
        patience = 0
        print('early stopping!')
        break
            


print('FairHGNN F1-sorce-HeBias:', "%0.2f%%" % (final_F1 * 100))
print('FairHGNN F1-mean-HeBias:', "%0.2f%%" % (final_F1_mean * 100))
print('FairHGNN F1-bias-HeBias:', "%0.2f%%" % (final_F1_bias * 100))