#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 09:29:03 2022

@author: Mesfin Diro
"""



import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops  #, softmax
from torch_geometric.nn.inits import zeros, glorot
from torch_geometric.typing import Adj
from torch_scatter import scatter_add
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils import softmax
from torch_geometric.graphgym.config import cfg
import torch.nn as nn



class MolGCNConv(MessagePassing):
    r""" General MolGCN Layer
    """
    def __init__(self, in_channels:int, out_channels:int, edge_dim:int, improved:bool=False, cached=False, bias:bool=True,**kwards):
        super(MolGCNConv, self).__init__(aggr='add', **kwards)  # "Add" aggregation.
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim # new
        self.improved = improved
        self.cached = cached
        self.normalize = cfg.gnn.normalize_adj

        self.weight = torch.nn.Parameter(torch.Tensor(in_channels, out_channels))
        nn.init.xavier_uniform_(self.weight.data, gain=1.414)
        self.edge_updated = torch.nn.Parameter(torch.Tensor(out_channels + edge_dim, out_channels))  # new property of GCN
        nn.init.xavier_uniform_(self.edge_updated.data, gain=1.414)
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.edge_updated)  # new
        zeros(self.bias)
        self.cached_result = None
        self.cached_num_edges = None

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight=None, improved=False,
             dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)

        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: Tensor,edge_weight=None):
        
        '''
        x : node feature matrix that has shape [N, in_channels]
        edge_index : connectivity, Adj list in the edge index has shape [2, E]
        edge_attr: N-dimensional edge feature matrix that has shape [ E x edge_dim]
        '''
        # Linearly transform node feature matrix  (XΘ)
        x = torch.matmul(x, self.weight)  

        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    'Cached {} number of edges, but found {}. Please '
                    'disable the caching behavior of this layer by removing '
                    'the `cached=True` argument in its constructor.'.format(
                        self.cached_num_edges, edge_index.size(1)))

        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            if self.normalize:
                edge_index, norm = self.norm(edge_index, x.size(self.node_dim),
                                             edge_weight, self.improved,
                                             x.dtype)
            else:
                norm = edge_weight
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result
        # Add node's self information (value=0) to edge_attr 
        self_loop_edges = torch.zeros(x.size(0), edge_attr.size(1)).to(edge_index.device)  # N x edge_dim   # new
        edge_attr = torch.cat([edge_attr, self_loop_edges], dim=0)  # new
        

        # Start propagating messages
        x_msg =  self.propagate(x=x, edge_index=edge_index, edge_attr=edge_attr, norm=norm)
        return x_msg
  

    def message(self, x_j, edge_attr, norm):   
        # Normalize node features (concat edge_attr)
        # x_j:  neighborhood
        if edge_attr is None:
            return norm.view(-1, 1) * x_j if norm is not None else x_j
        else:
            x_j = torch.cat([x_j, edge_attr], dim=-1)   # (N+E) x (emb(out)+edge_dim)   # new

            return norm.view(-1, 1) * x_j if norm is not None else x_j   
       

    def update(self, aggr_out):   #  Return node embeddings
        '''
        N x emb(out) = N x (emb(out)+edge_dim) @ (emb(out)+edge_dim) x emb(out)  
        For self Node  0(x_i): Based on the directed graph, Node 0 gets message from three edges and one self_loop
        For neighborhood Node(x_j):  only self_loop, since they do not get any message from others
        '''
        aggr_out = torch.mm(aggr_out, self.edge_updated)   # new property added

        if self.bias is not None:
            return aggr_out + self.bias
        else:
            return aggr_out
    def __repr__(self):
        return '{}({}, {},{})'.format(self.__class__.__name__, self.in_channels, self.out_channels,self.edge_dim)





class MolGATConv(MessagePassing):
    r""" The MolGAT convolutional operator for molecular graphs molecular data
    based on GAT model with attention mechanism with n-dimensional edge attr attributes
    Args:
       in_channels (int): Size of each input node feature.
       out_channels (int): Size of each output node feature.
       edge_dim (int): Size of each input edge feature.
       improved (bool, optional): Whether to use the improved GAT convolution from the 
           `"How to Find Your Friendly Neighborhood: Graph Attention Design with Self-Supervision" 
           <https://arxiv.org/abs/2010.14403>`_ paper. (default: False)
       cached (bool, optional): Whether to cache the computation for faster training. (default: False)
       heads (int, optional): Number of attention heads to use. (default: 1)
       negative_slope (float, optional): Negative slope coefficient for the leaky 
           rectified linear unit (LeakyReLU). (default: 0.2)
       dropout (float, optional): Dropout rate for the attention coefficients. (default: 0)
       bias (bool, optional): Whether to include a bias term. (default: True)
    """ 
    def __init__(self,
                 in_channels:int,
                 out_channels:int,
                 edge_dim:int,  # newly add 
                 improved:bool=False,
                 cached=False,
                 heads: int =1,
                 negative_slope:float=0.2,
                 dropout:float=0.,
                 bias:bool =True, **kwargs):
        super(MolGATConv, self).__init__(node_dim=0, aggr=cfg.gnn.agg, **kwargs)  # "Add" aggregation.

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim  # newly add
        self.heads = heads
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.improved = improved
        self.normalize = cfg.gnn.normalize_adj
        self.msg_direction = cfg.gnn.msg_direction
        
        
        self.weight = Parameter(torch.Tensor(in_channels, heads * out_channels))    # emb(in) x [H*emb(out)]
        nn.init.xavier_uniform_(self.weight.data, gain=1.414)
        self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels + edge_dim))   # 1 x H x [2*emb(out)+edge_dim]    # new
        nn.init.xavier_uniform_(self.att.data, gain=1.414)  

        if self.msg_direction == 'single':
            self.edge_updated = Parameter(torch.Tensor(out_channels + edge_dim, out_channels))   # [emb(out)+edge_dim] x emb(out)  # new
        else:
            self.edge_updated = Parameter(torch.Tensor(out_channels * 2 + edge_dim, out_channels))
            
   
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.att)
        glorot(self.edge_updated)  # new
        zeros(self.bias)
   
    def message(self, x_i, x_j, size_i, edge_index_i, edge_attr):  # Compute normalization (concatenate + softmax)
        '''
        x_i, x_j: after linear x and expand edge (N+E) x H x emb(out)
        = N x emb(in) @ emb(in) x [H*emb(out)] (+) E x H x emb(out)
        edge_index_i: the col part of index  [E+N]
        size_i: number of nodes
        edge_attr: edge values = (E+N) x edge_dim
        '''

        edge_attr = edge_attr.unsqueeze(1).repeat(1, self.heads, 1)  # (E+N) x H x edge_dim  # new

    
        
        # (E+N) x H x (emb(out)+edge_dim)   # new
        if self.msg_direction == 'both':  
            x_j = torch.cat((x_i, x_j, edge_attr), dim=-1)
        else:
            x_j = torch.cat((x_j, edge_attr), dim=-1)
 
        x_i = x_i.view(-1, self.heads, self.out_channels)  # (E+N) x H x emb(out)
        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)  # (E+N) x H

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, num_nodes=size_i)   # Computes a sparsely evaluated softmax

        if self.training and self.dropout > 0:
            alpha = F.dropout(alpha, p=self.dropout, training=True)
        xj_msg =  x_j* alpha.view(-1, self.heads, 1)
        return xj_msg

    def update(self, aggr_out):   
        '''
        # Return node embeddings (average heads)
        # for self Node 0(x_i): Based on the directed graph, Node 0 gets message from three edges and one self_loop
        # for neighborhood Node(x_j):  only self_loop, since they do not get any message from others
        '''
        aggr_out = aggr_out.mean(dim=1)
        aggr_out = torch.mm(aggr_out, self.edge_updated)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out
    def __repr__(self):
        return '{}({}, {}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.edge_dim, self.heads)

