#!/usr/bin/env python
# coding: utf-8
import math
import numpy as np
import torch
import copy
import pickle
import os
import time
import random
import scipy.sparse as sp
import torch.nn.functional as F
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

data_set='bitcoin'
# with open('/home/gjq/code_repository/dataset/dblp/adj_time_list.pickle', 'rb') as handle:
#     adj_time_list = pickle.load(handle, encoding='iso-8859-1')
# with open('/home/gjq/code_repository/CeDFormer/data/bitcoinotc/adj_time_list2000.pickle', 'rb') as handle:
#     adj_time_list = pickle.load(handle,encoding='iso-8859-1')
with open('/home/gjq/code_repository/DGCN-main/data/bitcoinotc/adj_time_list.pickle', 'rb') as handle:
    adj_time_list = pickle.load(handle,encoding='iso-8859-1')
    
# with open('/home/gjq/code_repository/DGCN-main/'+data_set+'/adj_orig_dense_list.pickle', 'rb') as handle:
#     adj_orig_dense_list = pickle.load(handle,encoding='bytes')

adj_orig_dense_list=[]
for i in adj_time_list:
    # print(torch.tensor(i.todense()).dtype)
    adj_orig_dense_list.append(torch.tensor(i.todense(),dtype=torch.float64))



def cosine_similarity(matrix1,matrix2):
    vector1=matrix1.view(-1)
    vector2=matrix2.view(-1)
    # 计算余弦相似度
    cosine = F.cosine_similarity(vector1, vector2, dim=0)

    return cosine

loss=np.empty([len(adj_orig_dense_list),len(adj_orig_dense_list)],dtype=np.float64)

print(adj_orig_dense_list[0].shape)
for i in range(len(adj_orig_dense_list)):
    for j in range(len(adj_orig_dense_list)):
        loss[i][j]=cosine_similarity(adj_orig_dense_list[i],adj_orig_dense_list[j])
print(loss)

# x_ticks = ['x-1', 'x-2', 'x-3']
# y_ticks = ['y-1', 'y-2', 'y-3']  # 自定义横纵轴
# ax = sns.heatmap(values, xticklabels=x_ticks, yticklabels=y_ticks)
mask = np.zeros_like(loss)
mask[np.triu_indices_from(mask)] = True
plt.legend(fontsize=20)
plt.subplots_adjust(bottom=5)
font = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 27,
        }
with sns.axes_style("white"):   
    ax = sns.heatmap(loss,mask=mask)
    #ax.set_title('Heatmap for Time Snapshot Dependency',fontsize=16)  # 图标题
    ax.set_xlabel('Time Snapshot',font)  # x轴标题
    ax.set_ylabel('Time Snapshot',font)
    plt.show()
    figure = ax.get_figure()
    figure.savefig('/home/gjq/code_repository/CeDFormer/data/'+data_set+'_sns_heatmap.jpg') 
    figure.savefig('/home/gjq/code_repository/CeDFormer/data/'+data_set+'_sns_heatmap.eps') 








