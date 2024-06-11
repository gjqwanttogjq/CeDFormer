#!/usr/bin/env python
# coding: utf-8

# importing libraries

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

 import numpy as np
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
import inspect
from sklearn.metrics import average_precision_score
import copy
import pickle
import os
import time
import random
from config import *
from utils import *
from embedding import *
from layer import *
from data_mask import *
from comm_discovery import *
device = torch.device('cuda:1')


# loading data

# with open('./data/highSchool/adj_time_list.pickle', 'rb') as handle:
#     adj_time_list = pickle.load(handle, encoding='iso-8859-1')
# with open('./data/highSchool/adj_orig_dense_list.pickle', 'rb') as handle:
#     adj_orig_dense_list = pickle.load(handle, encoding='bytes')

# with open("./data/ML-10m/graphs.npz", "rb") as f:
#     graphs = pickle.load(f)
# print("Loaded {} graphs ".format(len(graphs)))
# print(len(graphs))
# adj_time_list = [nx.adjacency_matrix(g) for g in graphs]
# adj_orig_dense_list = [nx.adjacency_matrix(g) for g in graphs]

adj_time_list=np.load("./data/ML-10m/graphs_1.npy", allow_pickle=True)

# When adj_orig_dense_list is too large, it can be calculated from adj_time_list
adj_orig_dense_list=[]
for i in adj_time_list:
    print(i)
    # print(torch.tensor(i.todense()).dtype)
    adj_orig_dense_list.append(torch.tensor(i.todense(),dtype=torch.float32))

# masking edges
outs = mask_edges_det(adj_time_list.to(device))
train_edges_l = outs[1]
pos_edges_l_n, false_edges_l_n = mask_edges_prd_new(
    adj_time_list, adj_orig_dense_list)
pos_edges_l, false_edges_l = mask_edges_prd(adj_time_list.to(device))



# creating edge list
edge_idx_list = []
for i in range(len(train_edges_l)):
    edge_idx_list.append(torch.tensor(
        np.transpose(train_edges_l[i]), dtype=torch.long))

class CeDFormer(nn.Module):
    def __init__(self, x_dim, embedding_dim, n_layers, ffn_hidden, n_head, eps, max_len, drop_prob, bias=False):
        super(CeDFormer, self).__init__() 
        self.x_dim = x_dim
        self.embedding_dim = embedding_dim

        self.n_layers = n_layers
        self.ffn_hidden = ffn_hidden

        self.eps = eps

        # embedding
        self.embedding = TransformerEmbedding(
            x_dim, embedding_dim, max_len, drop_prob)

        # transformer enc/dec
        self.encoders = nn.ModuleList([EncoderLayer(
            embedding_dim, ffn_hidden, n_head, drop_prob) for _ in range(n_layers)])
        self.decoders = nn.ModuleList([DecoderLayer(
            embedding_dim, ffn_hidden, n_head, drop_prob) for _ in range(n_layers)])
        self.dropout2 = nn.Dropout(p=drop_prob)
        self.norm2= LayerNorm(d_model=embedding_dim)
        self.joint=nn.Linear(embedding_dim+embedding_dim, embedding_dim)
        self.linear = nn.Linear(embedding_dim+embedding_dim, x_dim)
        self.linear2 = nn.Linear(embedding_dim, x_dim)
        self.gcn_emb=GCNConv(x_dim, embedding_dim)  
         
    def forward(self, x, comm_list, edge_idx_list, adj_orig_dense_list, emb_nodes_his=None,last_stamp_=None):
        assert len(adj_orig_dense_list) == len(edge_idx_list)

        kld_loss = 0
        nll_loss = 0
        x=x.to(cuda)
        # encoder-embedding
        # emb:[node,time,emb]
        emb,comm_emb,comm_num,comm_belong,emb_raw=self.embedding(x,edge_idx_list,center_flag,comm_list)

        num_dim,time_dim, emb_dim = emb.shape

        trg_mask = self.make_no_peak_mask(emb, emb).to(cuda)

        emb_all_nodes = []
        emb_all_comm = []
        emb_dec=[]

        if last_stamp_ is None:
            last_stamp = torch.zeros(num_dim, emb_dim).to(cuda)
            last_stamp.requires_grad = True
        else:
            last_stamp = last_stamp_[-1]

        if emb_nodes_his is None:
            # comm layer encoder
            for i in range(comm_num):
                emb_1 = comm_emb[i].clone()
                for encoder in self.encoders:
                    emb_1, kld_loss_node = encoder(emb_1)
                    kld_loss += kld_loss_node  # emb_1=Variable(emb_1)
                emb_all_comm.append(emb_1)
            # node layer encoder
            selected=set()
            for i in comm_list.keys():
                selected.update(comm_list[i])
            for i in range(num_dim):
                if i not in selected:
                    emb_1 = emb[i].clone()
                    for encoder in self.encoders:
                        emb_1, kld_loss_node = encoder(emb_1)
                        kld_loss += kld_loss_node  # emb_1=Variable(emb_1)
                    emb_all_nodes.append(emb_1)
                else:
                    emb_all_nodes.append(None)

            # emb_enc
            for i in range(num_dim):
                if emb_all_nodes[i] is None:
                    emb_1 = torch.empty(x.size(0), embedding_dim).to(cuda)
                    comm_result = find_key(comm_list, i)
                    if comm_result is not None:
                        node_comm=comm_result
                    else:
                        node_comm=-1
                    if node_comm != -1:
                        emb_1[:,:]=emb_all_comm[comm_belong[node_comm]][:]
                    emb_all_nodes[i]=emb_1
        else:        
            emb_all_nodes = emb_nodes_his

        # node layer decoder
        for i in range(num_dim):
            emb_1=emb_all_nodes[i]
            emb_temp = emb[i].clone()
            emb_temp=emb_temp[:-1]
            emb_last=emb_temp[-1]
            emb_2=torch.cat((last_stamp[i].unsqueeze(0),emb_temp),dim=0)
            for decoder in self.decoders:
                emb_2 = decoder(emb_2, emb_1, trg_mask)
            emb_dec.append(emb_2)
        emb_dec = torch.stack(emb_dec, dim=0).transpose(0, 1) #[time,node,emb]
        result=[]
        for t in range(time_dim):
            result.append(self.linear2(emb_dec[t]))
            nll_loss += self._nll_bernoulli(result[t], adj_orig_dense_list[t])
        return kld_loss, nll_loss,emb_all_nodes,emb_dec,result


    def _nll_bernoulli(self, logits, target_adj_dense):
        temp_size = target_adj_dense.size()[0]
        temp_sum = target_adj_dense.sum()
        posw = float(temp_size * temp_size - temp_sum) / temp_sum
        norm = temp_size * temp_size / \
            float((temp_size * temp_size - temp_sum) * 2)
        nll_loss_mat = F.binary_cross_entropy_with_logits(
            input=logits, target=target_adj_dense.to(cuda), pos_weight=posw, reduction='none')
        nll_loss = -1 * norm * torch.mean(nll_loss_mat, dim=[0, 1])
        return - nll_loss

    def make_no_peak_mask(self, q, k):
        len_q, len_k = q.size(1), k.size(1)
        # len_q x len_k
        mask = torch.tril(torch.ones(len_q, len_k)).type(torch.BoolTensor)

        return mask

    def make_prd_mask(self, q, k):
        len_q, len_k = q.size(1), k.size(1)

        # len_q x len_k
        mask = torch.tril(torch.ones(len_q, len_k), diagonal=-1).type(torch.BoolTensor)

        return mask

    def dec(self, z):
        outputs = InnerProductDecoder(act=lambda x: x)(z)
        return outputs

    def reset_parameters(self, stdv=1e-1):
        for weight in self.parameters():
            weight.data.normal_(0, stdv)

    def _init_weights(self, stdv):
        pass

seq_len = len(train_edges_l)
num_nodes = adj_orig_dense_list[seq_len-1].shape[0]
x_dim = num_nodes

comm_list=comm_discovery(num_nodes,adj_orig_dense_list,comm_rate)

# creating input tensors

x_in_list = []
for i in range(0, seq_len):
    x_temp = torch.tensor(np.eye(num_nodes).astype(np.float32))
    x_in_list.append(x_temp.clone().detach().requires_grad_(True))

x_in = Variable(torch.stack(x_in_list)).to(device)


# training
seq_start = 0
seq_end = seq_len - 3
tst_after = 0
early_stop=10
best_epoch_val_auc = 0
patient = 0
have_run=0
max_len = seq_len-3
total_time=0


# building model
model = CeDFormer(x_dim, embedding_dim, n_layers,
                      ffn_hidden, n_head, eps, max_len, drop_prob)
model.to(device)
cuda = next(model.parameters()).device
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for k in range(300):
    start_time = time.time()
    optimizer.zero_grad()
    kld_loss, nll_loss, nodes_emb ,last_stamp,_ = model(
        x_in[seq_start:seq_end], comm_list, edge_idx_list[seq_start:seq_end], adj_orig_dense_list[seq_start:seq_end])
    loss = nll_loss
    loss.backward()
    optimizer.step()
    nn.utils.clip_grad_norm_(model.parameters(), clip)
    end_time = time.time()
    epoch_time=end_time - start_time
    total_time+=epoch_time


    if k > tst_after:
        _, _,_, _,output = model(x_in[seq_end:seq_len], comm_list, edge_idx_list[seq_end:seq_len],
                                   adj_orig_dense_list[seq_end:seq_len],nodes_emb,last_stamp)
        auc_scores_prd, ap_scores_prd = get_roc_scores(
            pos_edges_l[seq_end:seq_len], false_edges_l[seq_end:seq_len], adj_orig_dense_list[seq_end:seq_len], output)

        auc_scores_prd_new, ap_scores_prd_new = get_roc_scores(
            pos_edges_l_n[seq_end:seq_len], false_edges_l_n[seq_end:seq_len], adj_orig_dense_list[seq_end:seq_len], output)
        if np.mean(np.array(auc_scores_prd)) > best_epoch_val_auc:
            best_epoch_val_auc = np.mean(np.array(auc_scores_prd))
            torch.save(model.state_dict(), "./model_checkpoints/model.pt")
            patient = 0
        else:
            patient += 1
            if patient > early_stop:
                have_run=k
                break
    print('epoch: ', k)
    print('epoch run time',epoch_time)
    print('nll_loss =', nll_loss.mean().item())
    print('loss =', loss.mean().item())
    if k > tst_after:
        print('----------------------------------')
        print('Link Prediction')
        print('link_prd_auc_mean', np.mean(np.array(auc_scores_prd)))
        print('link_prd_ap_mean', np.mean(np.array(ap_scores_prd)))
        print('----------------------------------')
        print('New Link Prediction')
        print('new_link_prd_auc_mean', np.mean(np.array(auc_scores_prd_new)))
        print('new_link_prd_ap_mean', np.mean(np.array(ap_scores_prd_new)))
        print('----------------------------------')
    print('----------------------------------')

run_time = total_time/have_run
print('average epoch run time',run_time)
