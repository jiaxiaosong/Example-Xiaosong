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
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        residual = x
        output = self.w_2(F.relu(self.w_1(x)))
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output


class BasicLayer(nn.Module):
    def __init__(self, input_dim, args):
        super(BasicLayer, self).__init__()
        ### Multi-head Atnn
        self.n_head = args.n_head
        self.act = torch.nn.functional.gelu
        #self.attn_dropot = dropout
        self.d_model = input_dim
        self.d_k = self.d_model // self.n_head
        self.d_v = self.d_model // self.n_head
        self.device = args.device
        self.inner_layer = args.inner_layer
        self.encoding = args.encoding
        self.w_qs = torch.nn.ModuleList([torch.nn.Linear(self.d_model, self.n_head * self.d_k, bias=False) for _ in range(self.inner_layer)])
        self.w_ks = torch.nn.ModuleList([torch.nn.Linear(self.d_model, self.n_head * self.d_k, bias=False) for _ in range(self.inner_layer)])
        self.w_vs = torch.nn.ModuleList([torch.nn.Linear(self.d_model, self.n_head * self.d_v, bias=False) for _ in range(self.inner_layer)])
        self.attention = ScaledDotProductAttention(temperature=np.power(self.d_k, 0.5), attn_dropout=args.dropout)        
        self.pos_ffn = torch.nn.ModuleList([PositionwiseFeedForward(self.d_model, self.d_model*4, dropout=args.dropout) for _ in range(self.inner_layer)])
        self.attn_fc = torch.nn.ModuleList([nn.Linear(self.n_head * self.d_v, self.d_model) for _ in range(self.inner_layer)])
        self.attn_layer_norm = torch.nn.ModuleList([nn.LayerNorm(self.d_model) for _ in range(self.inner_layer)])
        self.attn_dropout = torch.nn.ModuleList([nn.Dropout(args.dropout) for _ in range(self.inner_layer)])
        
        ##No Neighbor Vec
        self.no_neighbor_fc1 = nn.Linear(self.d_model//5*4, self.d_model//5*4)
        self.no_neighbor_fc2 = nn.Linear(self.d_model//5*4, self.d_model//5*4)
        self.no_neighbor_ln  = nn.LayerNorm(self.d_model//5*4)      
        nn.init.xavier_normal_(self.no_neighbor_fc1.weight)
        nn.init.xavier_normal_(self.no_neighbor_fc2.weight)

        for i in range(self.inner_layer):
            nn.init.normal_(self.w_qs[i].weight, mean=0, std=np.sqrt(2.0 / (self.d_model + self.d_k)))
            nn.init.normal_(self.w_ks[i].weight, mean=0, std=np.sqrt(2.0 / (self.d_model + self.d_k)))
            nn.init.normal_(self.w_vs[i].weight, mean=0, std=np.sqrt(2.0 / (self.d_model + self.d_v)))
            nn.init.xavier_normal_(self.attn_fc[i].weight)
        
        self.edge_embedding = torch.nn.Embedding(num_embeddings=3, embedding_dim=input_dim//5)

        ##Temporal Encoding
        self.basis_freq = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.d_model))).float()).view(1, -1).to(self.device)
        self.phase = torch.nn.Parameter(torch.zeros(self.d_model).float()).to(self.device)
        nn.init.normal_(self.phase)

        ###Aggregate Neighbor
        self.fea2node = nn.Linear(self.d_model, self.d_model//5*4)
        nn.init.xavier_normal_(self.fea2node.weight)
        self.layer_norm = nn.LayerNorm(self.d_model//5*4)

    def message_func(self, edges):
        z = torch.cat([edges.src['node_h'], self.edge_embedding(edges.data["etype"])], dim=-1)

        if self.encoding == "temporal":
            t_encoding = edges.dst['node_year'] - edges.src['node_year']
            t_encoding = t_encoding.unsqueeze(1) * self.basis_freq + self.phase
            t_encoding = torch.cos(t_encoding)
            z += t_encoding

        return {"z":z}

    def reduce_func(self, nodes):
        node_batch, neightbor_num, _ = nodes.mailbox['z'].size()
        if neightbor_num == 1:
            return  {"node_h":self.no_neighbor_ln(self.no_neighbor_fc2(self.act(self.no_neighbor_fc1(nodes.data["node_h"]))))}

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
    def __init__(self, input_dim, args):
        super(Model, self).__init__()
        self.n_layer = args.n_layer
        self.rawfeat_fc1 = nn.Linear(input_dim, input_dim*args.d_attn_ratio)
        self.rawfeat_fc2 = nn.Linear(input_dim*args.d_attn_ratio, input_dim*args.d_attn_ratio)
        self.rawfeat_ln = nn.LayerNorm(input_dim*args.d_attn_ratio)
        
        self.act = torch.nn.functional.gelu
        self.gnn_layers =  torch.nn.ModuleList([BasicLayer(input_dim*args.d_attn_ratio//4*5, args) for _ in range(self.n_layer)])
        self.device = args.device
    def forward(self, nf):
        nf.layers[0].data['node_h'] = self.rawfeat_ln(self.rawfeat_fc2(self.act(self.rawfeat_fc1(nf.layers[0].data['feat']))))
        for i in range(self.n_layer):
            nf.layers[i+1].data['node_h'] = nf.layers[i].data['node_h'][nf.map_from_parent_nid(layer_id=i, parent_nids=nf.layer_parent_nid(i+1), remap_local=True)]
            nf.block_compute(i, message_func=self.gnn_layers[i].message_func, reduce_func=self.gnn_layers[i].reduce_func)
        return nf.layers[-1].data.pop('node_h')

