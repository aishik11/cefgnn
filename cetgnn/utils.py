import random
from tqdm import tqdm
import dgl
from imports import device
import torch
from dgl.data import CoraGraphDataset
from dgl.nn import DeepWalk
from torch.optim import SparseAdam
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from dgl import RemoveSelfLoop

def prep_data(dataset, feat_key, emb_dim=32, walk_length=10, window_size=4):
    gs = []
    transform = RemoveSelfLoop()
    for g, l in dataset:
        g  = transform(g)
        gs.append(g.to(device))

    tbatch = dgl.batch(gs)

    model = DeepWalk(tbatch, emb_dim=emb_dim, walk_length=walk_length, window_size=window_size).to(device)
    dataloader = DataLoader(torch.arange(tbatch.num_nodes()), batch_size=128,
                            shuffle=True, collate_fn=model.sample)
    optimizer = SparseAdam(model.parameters(), lr=0.01)
    num_epochs = 5

    for epoch in range(num_epochs):
        for g in dataloader:
            loss = model(g)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    X = model.node_embed.weight.detach()
    z = torch.cat([X, tbatch.ndata[feat_key]], dim=-1)
    tbatch.ndata[feat_key] = z

    dataset_gs = dgl.unbatch(tbatch)

    moddataset = []

    for gm, (g, l) in zip(dataset_gs, dataset):
        moddataset.append((gm, l))

    dataset = moddataset
    return dataset


def get_supervised_dataloader(dataset, count, feat_key, labels):
    train_s_dataset = []
    for i in tqdm(range(count)):
        for g,l in dataset:
            ng = dgl.reorder_graph(g, node_permute_algo='rcmk')
            nsg = dgl.graph((ng.edges()[0], ng.edges()[1]), num_nodes=len(ng.nodes())).to(device)
            nsg.ndata['feat_onehot'] = ng.ndata[feat_key]
            train_s_dataset.append((nsg, labels.index(l)))
    return train_s_dataset
