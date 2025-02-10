import os

os.environ['DGLBACKEND'] = 'pytorch'
import torch
from dgl import RemoveSelfLoop
import dgl
from dgl import ToSimple
import torch.nn as nn
import numpy as np
device = 'cpu'

class edgepooling_training(nn.Module):

    def __init__(self, model, eps_in, n_class, merge_p=0.01, split_q=0.01, mult_fac=1, is_dgl_model=True, allow_disag=False):
        super(edgepooling_training, self).__init__()

        # self.transform = RemoveSelfLoop()
        self.transform1 = RemoveSelfLoop()
        self.is_dgl_model = is_dgl_model
        self.transform2 = ToSimple()
        self.allow_disag = allow_disag

        if allow_disag:
            self.score = self.score_allow_disag
        else:
            self.score = self.dont_allow_disag
        self.dropout = nn.Dropout(0.2)
        self.n_class = n_class
        self.model = model
        self.eps_in = eps_in
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.merge_p = merge_p
        self.split_q = split_q
        self.mult_fac = mult_fac

    def score_allow_disag(self,hstc, hdest, hcomb):
        return (2 + (hstc - hcomb)) * (
                2 + (hdest - hcomb))  # * ((1 + torch.floor((hstc - hcomb))) * (1+ torch.floor((hdest - hcomb))))

    def dont_allow_disag(self,hstc, hdest, hcomb):
        return (2 + (hstc - hcomb)) * (
                2 + (hdest - hcomb))   * ((1 + torch.floor((hstc - hcomb))) * (1+ torch.floor((hdest - hcomb))))
    def _get_activation_output(self, layer: nn.Module, x: torch.Tensor,
                               edge_index: torch.Tensor):
        """
        Get the activation of the layer.
        """
        activation = {}

        def get_activation():
            def hook(model, inp, out):
                activation['layer'] = out.detach()

            return hook

        layer.register_forward_hook(get_activation())

        with torch.no_grad():
            output = self.model(x, edge_index)

        return activation['layer'], output

    def get_score(self, bgraph, node_index_sum, score_map):
        key = str(node_index_sum.tolist())

        if key in score_map.keys():
            return score_map[key][0]
        else:
            x = [i for i in range(len(node_index_sum)) if node_index_sum[i] == 1]
            g = dgl.node_subgraph(bgraph, x)
            if self.is_dgl_model:
                batch = dgl.batch([g])
                e_sigmoids = self.model(batch, batch.ndata['feat_onehot'])
                score_map[key] = e_sigmoids[0]
                # TODO:implement return activations
            else:
                eid = torch.stack(g.edges())
                feat = g.ndata['feat_onehot']
                act, e_sigmoids = self._get_activation_output(list(self.model.modules())[-2], feat, eid)
                # e_sigmoids = self.model(feat, eid)
                score_map[key] = (e_sigmoids[0], torch.mean(act, 0).unsqueeze(0))
            return score_map[key][0]

    def pooling(self, bgraph, graph, nodefeat, prev_node_lvl_cluster, num_class, pool_it, score_map):

        edge_cross_class = list(range(len(graph.edges()[0])))
        edges = graph.edges()

        top_scores = []
        e_sigmoids = []
        node_index_sum = graph.ndata['node_index']
        for ni in node_index_sum:
            e_sigmoids.append(self.get_score(bgraph, ni, score_map))

        e_sigmoids = torch.stack(e_sigmoids).squeeze(-1)

        enodes = []
        for i in torch.softmax(e_sigmoids, dim=-1):
            enodes.append(i)
        enodes = torch.stack(enodes)

        # print(enodes)
        esrc = torch.transpose(enodes[edges[0]], 0, 1)
        edest = torch.transpose(enodes[edges[1]], 0, 1)

        node_index_sum = graph.ndata['node_index'][graph.edges()[0]] + graph.ndata['node_index'][graph.edges()[1]]

        ecomb_sigmoid = []

        node_count = []
        for ni in node_index_sum:
            ecomb_sigmoid.append(self.get_score(bgraph, ni, score_map))
            node_count.append(torch.sum(ni))

        ecomb_sigmoid = torch.stack(ecomb_sigmoid).squeeze(-1)

        ecomb = []
        for i in torch.softmax(ecomb_sigmoid, dim=-1):
            ecomb.append(i)

        ecomb = torch.transpose(torch.stack(ecomb), 0, 1)

        hstc = ((esrc[0] + 0.0000000001) * torch.log((1 / (esrc[0] + 0.0000000001)) + 0.0000000001) + (esrc[1] + 0.0000000001) * torch.log((1 / (esrc[1] + 0.0000000001)) + 0.0000000001)) * (
                           1 + (self.split_q / (1 + self.mult_fac * pool_it)))  # + (0.1/(pool_it+1))
        hdest = ((edest[0] + 0.0000000001) * torch.log((1 / (edest[0] + 0.0000000001)) + 0.0000000001) + (
                    edest[1] + 0.0000000001) * torch.log((1 / (edest[1] + 0.0000000001)) + 0.0000000001)) * (
                            1 + (self.split_q / (1 + self.mult_fac * pool_it)))  # + (0.1/(pool_it+1))
        hcomb = ((ecomb[0] + 0.0000000001) * torch.log((1 / (ecomb[0] + 0.0000000001)) + 0.0000000001) + (
                    ecomb[1] + 0.0000000001) * torch.log((1 / (ecomb[1] + 0.0000000001)) + 0.0000000001)) * (
                            1 + (self.merge_p / (1 + self.mult_fac * pool_it)))  # - (1/(pool_it+1))


        scores = self.score(hstc, hdest, hcomb)

        top_scores = top_scores + scores.tolist()

        top_scores = torch.tensor(top_scores)
        perm = torch.argsort(top_scores.squeeze(), descending=True).tolist()

        new_src = []
        new_dst = []
        mask = torch.zeros(len(graph.nodes()))#.to(device)
        new_nodes = []
        new_one_hot = []
        initial_c = max(prev_node_lvl_cluster)
        c = initial_c + 1
        cluster = [-1] * len(graph.nodes())
        selected_score = []
        for ex in perm:
            edge_index = edge_cross_class[ex]
            if scores[edge_index] <= 0:
                break

            node1 = graph.edges()[0][edge_index]
            node2 = graph.edges()[1][edge_index]

            if mask[node1] <= scores[edge_index] and mask[node2] <= scores[edge_index]:
                selected_score.append(ecomb_sigmoid[edge_index])
                node_feat_1 = nodefeat[node1]
                node_feat_2 = nodefeat[node2]
                if mask[node1] == 99999999:
                    print('annon----' + str(scores[edge_index]))
                if mask[node2] == 99999999:
                    print('annon----' + str(scores[edge_index]))
                mask[node1] = 99999999
                mask[node2] = 99999999

                new_nodes.append((node_feat_1 + node_feat_2))  # * e[edge_index])

                new_one_hot.append(graph.ndata['node_index'][node1] + graph.ndata['node_index'][node2])

                cluster[node1] = c
                cluster[node2] = c
                prev_node_lvl_cluster[prev_node_lvl_cluster == node1.item()] = c
                prev_node_lvl_cluster[prev_node_lvl_cluster == node2.item()] = c
                c += 1

            if mask[node1] < scores[edge_index]:
                mask[node1] = scores[edge_index]
            if mask[node2] < scores[edge_index]:
                mask[node2] = scores[edge_index]
        c = max(c, 0)
        for i in range(0, len(cluster)):
            if cluster[i] == -1:
                cluster[i] = c
                prev_node_lvl_cluster[prev_node_lvl_cluster == i] = c
                new_one_hot.append(graph.ndata['node_index'][i])
                new_nodes.append(nodefeat[i])
                c += 1

        cluster = cluster - initial_c - 1
        prev_node_lvl_cluster = prev_node_lvl_cluster - initial_c - 1

        for ei in range(len(edges[0])):
            new_src.append(cluster[graph.edges()[0][ei]])
            new_dst.append(cluster[graph.edges()[1][ei]])

        new_g = dgl.graph((list(new_src), list(new_dst)), num_nodes=len(new_nodes))
        new_g = new_g#.to(device)
        new_g.ndata["feat"] = torch.stack(new_nodes).double().to(device)
        new_g.edata['edge_score'] = ecomb_sigmoid
        new_g.ndata['node_index'] = torch.stack(new_one_hot)

        node_index_sum = new_g.ndata['node_index']  # + graph.ndata['node_index'][graph.edges()[1]]
        e_sigmoids = []
        for ni in node_index_sum:
            e_sigmoids.append(self.get_score(bgraph, ni, score_map))

        e_sigmoids = torch.stack(e_sigmoids).squeeze(-1)

        evotes = []
        for i in torch.softmax(e_sigmoids, dim=-1):
            ent = ((i[0]+ 0.0000000001) * torch.log((1 / (i[0]+ 0.0000000001))+ 0.0000000001) + (i[1]+ 0.0000000001) * torch.log((1 / (i[1]+ 0.0000000001))+ 0.0000000001))
            evotes.append(ent)

        evotes = torch.tensor(evotes)
        new_g.ndata['evotes'] = evotes

        return new_g, new_g.ndata["feat"], cluster, prev_node_lvl_cluster

    def forward(self, graph, h):

        graph.ndata['node_index'] = torch.eye(len(graph.nodes()))

        outs = []
        pooled_graph = graph

        nlclus = np.array(list(range(len(graph.nodes()))))

        nlclus_list = []
        pooled_graph_list = []
        pcluster_list = []

        last_nodes = pooled_graph.nodes()
        pooled_graph_features = h.float()

        pooled_graph = self.transform1(pooled_graph)

        pc = 0
        score_map = {}
        while True:
            if pooled_graph.num_edges() == 0:
                break;
            pooled_graph, pooled_graph_features, clusters, nlclus = self.pooling(graph, pooled_graph,
                                                                                 pooled_graph_features.float(),
                                                                                 nlclus, self.n_class, pc,
                                                                                 score_map)
            pc += 1

            pooled_graph = self.transform1(pooled_graph)
            if len(pooled_graph.nodes()) == len(last_nodes) and len(pcluster_list) > 0:
                break
            else:
                nlclus_list.append(nlclus.copy())

                pooled_graph_list.append(pooled_graph)
                pcluster_list.append(clusters.copy())
                last_nodes = pooled_graph.nodes()

        return outs, nlclus_list, pcluster_list, pooled_graph_list
