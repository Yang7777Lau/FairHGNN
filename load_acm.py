import torch
import numpy as np
import scipy.io as sio
from torch_geometric.data import HeteroData
from dgl.data import DGLDataset
from dgl.data.utils import download, get_download_dir, _get_dgl_url
import dgl
from torch_geometric.data import InMemoryDataset, HeteroData
import torch_geometric as pyg
from torch_geometric.utils import from_dgl
import os
import os.path as osp
from itertools import product
from typing import Callable, List, Optional
import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.nn import MetaPath2Vec
from torch_geometric.transforms import AddMetaPaths
import argparse

parser = argparse.ArgumentParser(description='dblp')
parser.add_argument('--seed', type=int, default=89)
parser.add_argument('--dataset', type=str, default='DBLP')
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--gamma', type=float, default=0.4)
parser.add_argument('--alhpa', type=float, default=0.4)
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--model', type=str, default='HAN')
parser.add_argument('--temperature', type=float, default='1.5')
args = parser.parse_args()


device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
def get_binary_mask(total_size, indices):
    mask = torch.zeros(total_size)
    mask[indices] = 1
    return mask.byte()
remove_self_loop=False

def load_acm_raw(remove_self_loop):
    assert not remove_self_loop
    url = 'data/ACM.mat'
    data_path = 'data/ACM.mat'


    data = sio.loadmat(data_path)
    p_vs_l = data['PvsL']       # paper-field?
    p_vs_a = data['PvsA']       # paper-author
    p_vs_t = data['PvsT']       # paper-term, bag of words
    p_vs_c = data['PvsC']       # paper-conference, labels come from that

    # We assign
    # (1) KDD papers as class 0 (data mining),
    # (2) SIGMOD and VLDB papers as class 1 (database),
    # (3) SIGCOMM and MOBICOMM papers as class 2 (communication)
    conf_ids = [0, 1, 9, 10, 13]
    label_ids = [0, 1, 2, 2, 1]

    p_vs_c_filter = p_vs_c[:, conf_ids]
    p_selected = (p_vs_c_filter.sum(1) != 0).A1.nonzero()[0]
    p_vs_l = p_vs_l[p_selected]
    p_vs_a = p_vs_a[p_selected]
    p_vs_t = p_vs_t[p_selected]
    p_vs_c = p_vs_c[p_selected]

    hg = dgl.heterograph({
        ('paper', 'pa', 'author'): p_vs_a.nonzero(),
        ('author', 'ap', 'paper'): p_vs_a.transpose().nonzero(),
        ('paper', 'pf', 'field'): p_vs_l.nonzero(),
        ('field', 'fp', 'paper'): p_vs_l.transpose().nonzero()
    })

    features = torch.FloatTensor(p_vs_t.toarray())

    pc_p, pc_c = p_vs_c.nonzero()
    labels = np.zeros(len(p_selected), dtype=np.int64)
    for conf_id, label_id in zip(conf_ids, label_ids):
        labels[pc_p[pc_c == conf_id]] = label_id
    labels = torch.LongTensor(labels)

    
    num_classes = 3

    float_mask = np.zeros(len(pc_p))
    for conf_id in conf_ids:
        pc_c_mask = (pc_c == conf_id)
        float_mask[pc_c_mask] = np.random.permutation(np.linspace(0, 1, pc_c_mask.sum()))
    train_idx = np.where(float_mask <= 0.2)[0]
    val_idx = np.where((float_mask > 0.2) & (float_mask <= 0.3))[0]
    test_idx = np.where(float_mask > 0.3)[0]

    num_nodes = hg.number_of_nodes('paper')

    train_mask = get_binary_mask(num_nodes, train_idx)
    val_mask = get_binary_mask(num_nodes, val_idx)
    test_mask = get_binary_mask(num_nodes, test_idx)


    hg.nodes['paper'].data['x'] = features
    hg.nodes['paper'].data['y'] = labels
    hg.nodes['paper'].data['train_mask'] = train_mask
    hg.nodes['paper'].data['val_mask'] = val_mask
    hg.nodes['paper'].data['test_mask'] = test_mask
    hg.nodes['paper'].data['test_mask'] = test_mask


    print(hg.ntypes)
    graph = from_dgl(hg)
    graph['author'].num_nodes=hg.num_nodes('author')
    graph['field'].num_nodes=hg.num_nodes('field')


    '''
    metapath = [
        ('author', 'ap', 'paper'),
        ('paper', 'pf', 'field'),
        ('field', 'fp', 'paper'),
        ('paper', 'pa', 'author'),

    ]

    # 初始化MetaPath2Vec模型
    model = MetaPath2Vec(
        graph.edge_index_dict,  # 图的边索引
        embedding_dim=128,  # 嵌入维度
        metapath=metapath,  # 定义的元路径
        walk_length=25,  # 游走长度
        context_size=10,  # 上下文窗口大小
        walks_per_node=5, 
        num_negative_samples=5,
        sparse=True  
    ).to(device)  # 选择运行设备

    loader = model.loader(batch_size=128, shuffle=True, num_workers=6)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)


    def train(epoch, log_steps=100, eval_steps=2000):
        model.train()

        total_loss = 0
        for i, (pos_rw, neg_rw) in enumerate(loader):
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if (i + 1) % log_steps == 0:
                print((f'Epoch: {epoch}, Step: {i + 1:05d}/{len(loader)}, '
                       f'Loss: {total_loss / log_steps:.4f}'))
                total_loss = 0


    # 训练一定数量的轮次
    num_epochs = 20
    for epoch in range(num_epochs):
        train(epoch)
    model.eval()    
    embedding_author = model('author')  
    embedding_field = model('field') 
    graph['author'].x=embedding_author
    graph['field'].x=embedding_field
    '''
    graph['author'].x = torch.ones((graph['author'].num_nodes, 1))
    graph['field'].x = torch.ones((graph['field'].num_nodes, 1))
    print(graph)
    return graph  ,num_nodes 
#,train_mask ,val_mask,test_mask

