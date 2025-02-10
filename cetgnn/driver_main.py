import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, GCNConv
from torch_geometric.data import Data
from torch_geometric.utils import convert
from graphxai.datasets import FluorideCarbonyl
from graphxai.explainers import GNNExplainer, PGExplainer, SubgraphX
from graphxai.utils import Explanation, EnclosingSubgraph
import graphxai.metrics.metrics_graph as mat
from dgl import RemoveSelfLoop
import numpy as np
import random
from tqdm import tqdm
import os

# Import your custom modules
import utils
import models

if torch.cuda.is_available():
  device = 'cuda'
else:
  device = 'cpu'
# Set random seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# Define the GCN model
class GCN_6layer(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN_6layer, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv4 = GCNConv(hidden_channels, hidden_channels)
        self.conv5 = GCNConv(hidden_channels, hidden_channels)
        self.conv6 = GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch=None):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index).relu()
        x = self.conv4(x, edge_index).relu()
        x = self.conv5(x, edge_index).relu()
        x = self.conv6(x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.lin(x)
        return x


def process_dataloader(data_list):
    datam = []
    dw_dim, dw_walk_length, dw_window_size = 16, 6, 3
    feat_key = 'attr'

    list_exp = []
    for g, exp in zip(data_list[0], data_list[1]):
        dgg = convert.to_dgl(g)
        dgg.ndata['attr'] = dgg.ndata['x']
        dgg.ndata['feat_onehot'] = dgg.ndata['attr']
        datam.append((dgg, g.y.item()))
        list_exp.append(exp)

    data = utils.prep_data(datam, feat_key, dw_dim, dw_walk_length, dw_window_size)

    pyg_data = []
    for gl, ex in zip(data, list_exp):
        g, l = gl
        pyg_data.append(
            Data(x=g.ndata['attr'].to(device), edge_index=torch.stack(g.edges()).to(device), y=torch.tensor([l])).to(
                device))

    return data, pyg_data, list_exp


def train_model(model, train_pyg_data, test_pyg_data, num_epochs=400):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    best_model = None
    lowest_loss = float('inf')

    for epoch in tqdm(range(num_epochs)):
        model.train()
        for d in train_pyg_data[1]:
            optimizer.zero_grad()
            out = model(d.x, d.edge_index, None)
            loss = criterion(F.log_softmax(out, dim=-1), d.y)
            loss.backward()
            optimizer.step()

        model.eval()
        test_loss = 0
        with torch.no_grad():
            for d in test_pyg_data[1]:
                out = model(d.x, d.edge_index, None)
                test_loss += criterion(F.softmax(out, dim=-1), d.y).item()

        avg_test_loss = test_loss / len(test_pyg_data[1])
        if avg_test_loss < lowest_loss:
            lowest_loss = avg_test_loss
            best_model = model.state_dict()

    model.load_state_dict(best_model)
    return model


def run_our_framework(model, test_pyg_data):
    model.eval()
    model.to('cpu')
    pooler = models.edgepooling_training(model, 0.9, 2, is_dgl_model=False, merge_p=0.000001, split_q=0.4, mult_fac=11,
                                         allow_disag=False)

    out_exp = []
    gt_exps = []
    data = []

    for dgg, pyg, gexp in tqdm(list(zip(test_pyg_data[0], test_pyg_data[1], test_pyg_data[2]))[:100]):
        g, _ = dgg
        if pyg.y.item() == 0:
            continue

        g = g.cpu()
        g.ndata['feat_onehot'] = g.ndata['attr']
        outs, nlclus_list, pcluster_list, pooled_graph_list = pooler(g, g.ndata['attr'])

        final_set = []
        with torch.no_grad():
            for pg in pooled_graph_list[:]:
                mdata = []
                for ni, ev in zip(pg.ndata['node_index'], pg.ndata['evotes']):
                    nodes_in = [i for i in range(len(g.nodes())) if ni[i] != 0]
                    mdata.append({'nodes': nodes_in, 'evote': ev})
                final_set = mdata
                if len(mdata) < 3:
                    break

        lowest = min(m['evote'].item() for m in final_set)
        node_imps = [1 if any(i in m['nodes'] and m['evote'].item() <= lowest for m in final_set) else 0 for i in
                     range(len(g.nodes()))]

        node_imps = torch.tensor(node_imps)
        exp = Explanation(node_imp=node_imps)
        sub = EnclosingSubgraph(nodes=node_imps, edge_index=pyg.edge_index, inv=None, edge_mask=None)
        exp.set_enclosing_subgraph(sub)
        exp.set_whole_graph(Data(x=g.ndata['attr'], edge_index=torch.stack(g.edges())))

        data.append(pyg)
        out_exp.append(exp)
        gt_exps.append(gexp)

    return out_exp, gt_exps, data


def run_subgraphx(model, test_pyg_data):
    subx = SubgraphX(model, reward_method='gnn_score', num_hops=3, rollout=5)
    sub_exp = []
    gt_exps = []

    for pyg, gexp in tqdm(list(zip(test_pyg_data[1], test_pyg_data[2]))[:100]):
        if pyg.y.item() == 0:
            continue
        exp = subx.get_explanation_graph(x=pyg.x.to(device), edge_index=pyg.edge_index.to(device), max_nodes=14)
        sub_exp.append(exp)
        gt_exps.append(gexp)

    return sub_exp, gt_exps


def run_gnnexplainer(model, test_pyg_data):
    explainer = GNNExplainer(model)
    gnn_exp = []
    gt_exps = []

    null_batch = torch.zeros(1).long()
    forward_kwargs = {'batch': null_batch}

    for pyg, gexp in tqdm(list(zip(test_pyg_data[1], test_pyg_data[2]))[:100]):
        exp = explainer.get_explanation_graph(x=pyg.x, edge_index=pyg.edge_index, forward_kwargs=forward_kwargs)
        gnn_exp.append(exp)
        gt_exps.append(gexp)

    return gnn_exp, gt_exps


def run_pgexplainer(model, train_pyg_data, test_pyg_data):
    null_batch = torch.zeros(1).long().to(device)
    forward_kwargs = {'batch': null_batch}
    explainer = PGExplainer(model, emb_layer_name='conv6', max_epochs=10, lr=0.1, explain_graph=True)

    tot_data = train_pyg_data[1] + train_pyg_data[1]
    tot_data = [x.to(device) for x in tot_data]
    explainer.train_explanation_model(tot_data, forward_kwargs=forward_kwargs)

    pge_exp = []
    gt_exps = []

    for pyg, gexp in tqdm(list(zip(test_pyg_data[1], test_pyg_data[2]))[:100]):
        exp = explainer.get_explanation_graph(x=pyg.x, edge_index=pyg.edge_index, label=pyg.y,
                                              forward_kwargs=forward_kwargs)
        pge_exp.append(exp)
        gt_exps.append(gexp)

    return pge_exp, gt_exps


def evaluate_explanations(out_exp, gt_exps, data, model):
    JACs = []
    out_mat = []

    for ox, gtx, d in zip(out_exp, gt_exps, data):
        total_gt = sum(gt.node_imp for gt in gtx)
        total_out = ox.node_imp

        TP = sum((p == 1 and l == 1) for p, l in zip(total_out, total_gt))
        FP = sum((p == 1 and l == 0) for p, l in zip(total_out, total_gt))
        FN = sum((p == 0 and l == 1) for p, l in zip(total_out, total_gt))

        JAC = TP / (TP + FP + FN + 1e-09)
        JACs.append(JAC)

        faith = mat.graph_exp_faith_graph(ox, d, model)[1]
        out_mat.append(faith)

    return {
        'jac': np.mean(JACs),
        'jac_std': np.std(JACs) / np.sqrt(len(JACs)),
        'faith': np.mean(out_mat),
        'faith_std': np.std(out_mat) / np.sqrt(len(out_mat))
    }


def main():
    set_seed(42)

    # Initialize dataset
    dataset = FluorideCarbonyl(split_sizes=(0.8, 0.2, 0), seed=42)

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Process data
    train_pyg_data = process_dataloader(dataset.get_train_list())
    test_pyg_data = process_dataloader(dataset.get_test_list())

    # Initialize and train model
    model = GCN_6layer(30, 32, 2).to(device)
    model = train_model(model, train_pyg_data, test_pyg_data)

    # Run explainers
    results = []

    # Our framework
    print("Running our framework...")
    our_exp, our_gt_exps, our_data = run_our_framework(model, test_pyg_data)
    our_results = evaluate_explanations(our_exp, our_gt_exps, our_data, model)
    results.append({'model': 'ours', **our_results})

    # SubgraphX
    print("Running SubgraphX...")
    sub_exp, sub_gt_exps = run_subgraphx(model, test_pyg_data)
    sub_results = evaluate_explanations(sub_exp, sub_gt_exps, our_data, model)
    results.append({'model': 'subgraphx', **sub_results})

    # GNNExplainer
    print("Running GNNExplainer...")
    gnn_exp, gnn_gt_exps = run_gnnexplainer(model, test_pyg_data)
    gnn_results = evaluate_explanations(gnn_exp, gnn_gt_exps, our_data, model)
    results.append({'model': 'gnnexplainer', **gnn_results})

    # PGExplainer
    print("Running PGExplainer...")
    pge_exp, pge_gt_exps = run_pgexplainer(model, train_pyg_data, test_pyg_data)
    pge_results = evaluate_explanations(pge_exp, pge_gt_exps, our_data, model)
    results.append({'model': 'pgexplainer', **pge_results})

    # Save results
    df = pd.DataFrame(results)
    df.to_csv("explainer_comparison_results.csv", index=False)
    print("Results saved to explainer_comparison_results.csv")


if __name__ == "__main__":
    main()