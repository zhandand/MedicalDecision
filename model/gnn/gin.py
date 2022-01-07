import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.glob import (AvgPooling, GlobalAttentionPooling,
                                 MaxPooling, SumPooling)


class GINLayer(nn.Module):
    r"""Single Layer GIN from `Strategies for
    Pre-training Graph Neural Networks <https://arxiv.org/abs/1905.12265>`__

    Parameters
    ----------
    num_edge_emb_list : list of int
        num_edge_emb_list[i] gives the number of items to embed for the
        i-th categorical edge feature variables. E.g. num_edge_emb_list[0] can be
        the number of bond types and num_edge_emb_list[1] can be the number of
        bond direction types.
    emb_dim : int
        The size of each embedding vector.
    batch_norm : bool
        Whether to apply batch normalization to the output of message passing.
        Default to True.
    activation : None or callable
        Activation function to apply to the output node representations.
        Default to None.
    """

    def __init__(self, emb_dim, activation=None):
        super(GINLayer, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 2 * emb_dim),
            nn.ReLU(),
            nn.Linear(2 * emb_dim, emb_dim)
        )
        self.bn = nn.BatchNorm1d(emb_dim)
        self.activation = activation
        self.reset_parameters()
        self.eps = torch.nn.Parameter(torch.FloatTensor([0]))

    def reset_parameters(self):
        """Reinitialize model parameters."""
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()

        if self.bn is not None:
            self.bn.reset_parameters()

    def forward(self, g, node_feats):
        """Update node representations.

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs
        node_feats : FloatTensor of shape (N, emb_dim)
            * Input node features
            * N is the total number of nodes in the batch of graphs
            * emb_dim is the input node feature size, which must match emb_dim in initialization
        categorical_edge_feats : list of LongTensor of shape (E)
            * Input categorical edge features
            * len(categorical_edge_feats) should be the same as len(self.edge_embeddings)
            * E is the total number of edges in the batch of graphs

        Returns
        -------
        node_feats : float32 tensor of shape (N, emb_dim)
            Output node representations
        """

        g = g.local_var()
        g.ndata['feat'] = node_feats
        g.update_all(fn.copy_u('feat', 'm'), fn.sum('m', 'neigh'))
        node_feats = (1 + self.eps) * g.ndata['feat'] + g.ndata['neigh']
        node_feats = self.bn(self.mlp(node_feats))
        if self.activation is not None:
            node_feats = self.activation(node_feats)
        return node_feats


class GINEncoder(nn.Module):
    def __init__(self, num_layers=5,
                 emb_dim=300, JK='last', dropout=0.5, readout='mean', n_tasks=1):
        super(GINEncoder, self).__init__()

        self.num_layers = num_layers
        self.JK = JK
        self.dropout = nn.Dropout(dropout)

        if num_layers < 2:
            raise ValueError('Number of GNN layers must be greater '
                             'than 1, got {:d}'.format(num_layers))

        self.gnn_layers = nn.ModuleList()
        for layer in range(num_layers):
            if layer == num_layers - 1:
                self.gnn_layers.append(GINLayer(emb_dim))
            else:
                self.gnn_layers.append(GINLayer(emb_dim, activation=F.relu))

        if readout == 'sum':
            self.readout = SumPooling()
        elif readout == 'mean':
            self.readout = AvgPooling()
        elif readout == 'max':
            self.readout = MaxPooling()
        elif readout == 'attention':
            if JK == 'concat':
                self.readout = GlobalAttentionPooling(
                    gate_nn=nn.Linear((num_layers + 1) * emb_dim, 1))
            else:
                self.readout = GlobalAttentionPooling(
                    gate_nn=nn.Linear(emb_dim, 1))
        else:
            raise ValueError("Expect readout to be 'sum', 'mean', "
                             "'max' or 'attention', got {}".format(readout))
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize model parameters."""

        for layer in self.gnn_layers:
            layer.reset_parameters()

    def forward(self, g, node_feats):
        all_layer_node_feats = [node_feats]
        for layer in range(self.num_layers):
            node_feats = self.gnn_layers[layer](g, all_layer_node_feats[layer])
            node_feats = self.dropout(node_feats)
            all_layer_node_feats.append(node_feats)

        if self.JK == 'concat':
            final_node_feats = torch.cat(all_layer_node_feats, dim=1)
        elif self.JK == 'last':
            final_node_feats = all_layer_node_feats[-1]
        elif self.JK == 'max':
            all_layer_node_feats = [h.unsqueeze(
                0) for h in all_layer_node_feats]
            final_node_feats = torch.max(
                torch.cat(all_layer_node_feats, dim=0), dim=0)[0]
        elif self.JK == 'sum':
            all_layer_node_feats = [h.unsqueeze(
                0) for h in all_layer_node_feats]
            final_node_feats = torch.sum(
                torch.cat(all_layer_node_feats, dim=0), dim=0)
        else:
            return ValueError("Expect self.JK to be 'concat', 'last', "
                              "'max' or 'sum', got {}".format(self.JK))

        graph_feats = self.readout(g, final_node_feats)
        return graph_feats
