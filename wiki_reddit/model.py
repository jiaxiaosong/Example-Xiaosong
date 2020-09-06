import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph


class ScaledDotProductAttention(torch.nn.Module):
    """ Scaled Dot-Product Attention """
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = torch.nn.Dropout(attn_dropout)
        self.softmax = torch.nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -1e10)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''
    def __init__(self, d_in, d_hid, dropout):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1) # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, 1) # position-wise
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output


class BasicLayer(nn.Module):
    def __init__(self, n_head, node_dim, d_k, d_v, args, edge_dim=None, dropout=0.1, act=torch.nn.functional.gelu, device="cpu"):
        super(BasicLayer, self).__init__()
        ### Multi-head Atnn
        self.n_head = n_head
        self.act = act
        self.d_k = d_k
        self.d_v = d_v
        self.edge_dim = edge_dim
        self.node_dim = node_dim
        self.device = device
        self.encoding = args.encoding
        self.inner_layer = args.inner_layer

        d_model = node_dim
        if  self.edge_dim:
            self.edge_fc = nn.Linear(self.edge_dim, self.node_dim)
            nn.init.xavier_normal_(self.edge_fc.weight)
        self.d_model = d_model

        self.w_qs = torch.nn.ModuleList([torch.nn.Linear(d_model, n_head * d_k, bias=False) for _ in range(self.inner_layer)])
        self.w_ks = torch.nn.ModuleList([torch.nn.Linear(d_model, n_head * d_k, bias=False) for _ in range(self.inner_layer)])
        self.w_vs = torch.nn.ModuleList([torch.nn.Linear(d_model, n_head * d_v, bias=False) for _ in range(self.inner_layer)])
        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5), attn_dropout=dropout)        
        self.pos_ffn = torch.nn.ModuleList([PositionwiseFeedForward(d_model, d_model*4, dropout=dropout) for _ in range(self.inner_layer)])
        self.attn_fc = torch.nn.ModuleList([nn.Linear(n_head * d_v, d_model) for _ in range(self.inner_layer)])
        self.attn_layer_norm = torch.nn.ModuleList([nn.LayerNorm(d_model) for _ in range(self.inner_layer)])
        self.attn_dropout = torch.nn.ModuleList([nn.Dropout(dropout) for _ in range(self.inner_layer)])
        for i in range(self.inner_layer):
            nn.init.normal_(self.w_qs[i].weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
            nn.init.normal_(self.w_ks[i].weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
            nn.init.normal_(self.w_vs[i].weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))
            nn.init.xavier_normal_(self.attn_fc[i].weight)

        ##None Time-step
        self.non_vec = torch.nn.Parameter(torch.zeros(1, self.node_dim).float()).to(device)
        nn.init.normal_(self.non_vec)
        ##Temporal Encoding
        self.basis_freq = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.d_model))).float()).view(1, -1).to(device)
        self.phase = torch.nn.Parameter(torch.zeros(self.d_model).float()).to(device)
        nn.init.normal_(self.phase)
        self.t_now = None

        ###Aggregate Neighbor
        self.fea2node = nn.Linear(d_model, self.node_dim)
        nn.init.xavier_normal_(self.fea2node.weight)
        self.layer_norm = nn.LayerNorm(self.node_dim)

    def message_func(self, edges):
        if self.encoding == "temporal":
            #edge_features: time-stamp
            t_encoding = self.t_now - edges.data["t"]#Edge_Num, 1
            t_encoding = t_encoding * self.basis_freq + self.phase #edge_batch, d_t
            t_encoding = torch.cos(t_encoding)
        if self.encoding == 'none':
            t_encoding = torch.zeros(edges.src['node_h'].shape[0], self.d_model).to(self.device)
        if self.edge_dim:
            z = edges.src['node_h'] + self.act(self.edge_fc(edges.data["edge_raw_feat"])) + t_encoding #edge_batch, d_model (d_model = n_head*d_v)
        else:
            z = edges.src['node_h'] + t_encoding#edge_batch, d_model (d_model = n_head*d_v)
        return {"z":z}

    def reduce_func(self, nodes):
        node_batch, neightbor_num, _ = nodes.mailbox['z'].size()
        if neightbor_num == 1:
            return  {"node_h":self.non_vec.repeat(node_batch, 1)}
        f_in = nodes.mailbox['z']
        for i in range(self.inner_layer):
            q = self.w_qs[i](f_in).view(node_batch, neightbor_num, self.n_head, -1).permute(2, 0, 1, 3).contiguous().view(-1, neightbor_num, self.d_k) # (n_head*node_batch), neightbor_num, d_k
            k = self.w_ks[i](f_in).view(node_batch, neightbor_num, self.n_head, -1).permute(2, 0, 1, 3).contiguous().view(-1, neightbor_num, self.d_k) # (n_head*node_batch), neightbor_num, d_k
            v = self.w_vs[i](f_in).view(node_batch, neightbor_num, self.n_head, -1).permute(2, 0, 1, 3).contiguous().view(-1, neightbor_num, self.d_v) # (n_head*node_batch), neightbor_num, d_v

            output, attn = self.attention(q, k, v, mask=None) #(n_head*node_batch), neighbor_num, d_v
            output = output.view(self.n_head, node_batch, neightbor_num, -1).permute(1, 2, 0, 3).contiguous().view(node_batch, neightbor_num, self.n_head*self.d_v)  #node_batch, neighbor_num, d_v * n_head
            output = self.attn_dropout[i](self.attn_fc[i](output)) #node_batch, neighbor_num, d_model
            output = self.attn_layer_norm[i](output + f_in)
            output = self.pos_ffn[i](output) #node_batch, neighbor_num, d_model
            f_in = output
        output = output.mean(dim=1) #node_batch, d_model
        output = self.layer_norm(self.act(self.fea2node(output)) + nodes.data["node_h"])
        return {"node_h":output}

class Model(nn.Module):
    def __init__(self, num_layers, n_head, node_dim, d_k, d_v, d_T, args, edge_dim = None, dropout=0.1, act=torch.nn.functional.gelu, device="cpu"):
        super(Model, self).__init__()
        self.gnn_layers =  torch.nn.ModuleList([BasicLayer(n_head, node_dim, d_k, d_v, args, edge_dim, dropout, act, device=device) for _ in range(num_layers)])
        self.self_loop_embedding = torch.nn.Parameter(torch.zeros(edge_dim)).unsqueeze(0).to(device)
        nn.init.normal_(self.self_loop_embedding)
        self.num_layers = num_layers
        self.device = device
    def forward(self, nf, t_now):
        nf.layers[0].data['node_h'] = nf.layers[0].data['node_raw_feat']
        for i in range(self.num_layers):
            self_loop_edges = (nf.map_to_parent_nid(nf.block_edges(i)[0])==nf.map_to_parent_nid(nf.block_edges(i)[1])).nonzero().squeeze().type(torch.int64)
            nf.blocks[i].data["edge_raw_feat"][self_loop_edges] = self.self_loop_embedding.repeat(self_loop_edges.shape[0], 1)
            self.gnn_layers[i].t_now = t_now
            ##Self_loop Term
            nf.layers[i+1].data['node_h'] = nf.layers[i].data['node_h'][nf.map_from_parent_nid(layer_id=i, parent_nids=nf.layer_parent_nid(i+1),  remap_local=True)]
            nf.block_compute(i, message_func=self.gnn_layers[i].message_func, reduce_func=self.gnn_layers[i].reduce_func)
        
        return nf.layers[-1].data.pop('node_h')


