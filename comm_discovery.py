import networkx as nx
import torch
import numpy as np

def cal_conductance(graph,node_set):
    internal_edges = 0
    external_edges = 0

    for node in node_set:
        neighbors = set(graph.neighbors(node))
        internal_edges += len(neighbors.intersection(node_set))
        external_edges += len(neighbors.difference(node_set))

    conductance = external_edges / (2 * internal_edges + external_edges)
    return conductance

def comm_discovery(num_nodes,adj_orig_dense_list,p):
    spatial_graph=torch.zeros(num_nodes,num_nodes)
    for i in range(len(adj_orig_dense_list)):
        spatial_graph=spatial_graph+adj_orig_dense_list[i]
    # spatial_graph[spatial_graph>1]=1

    G = nx.Graph(np.array(spatial_graph))
    pagerank_scores = nx.pagerank(G)
    items=list(pagerank_scores.items())
    items.sort(key=lambda x:x[1],reverse=True)
    pagerank_scores=[(key,value)for key,value in items]
    center_nodes=[i[0] for i in pagerank_scores ]
    nodes=G.nodes()

    selected_node=0
    comm_list={}
    end_flag=0

    while True:
        if end_flag>p*len(pagerank_scores):
            break
        if selected_node>=len(center_nodes):
            break
        if center_nodes[selected_node] not in nodes:
            selected_node+=1
            continue
        if len(set(G.neighbors(center_nodes[selected_node])))==0:
            selected_node+=1
            continue
        center_node=center_nodes[selected_node]
        best_comm=set([center_nodes[selected_node]])
        best_conductance=cal_conductance(G,best_comm)
        for node in best_comm:
            current_comm=best_comm|set(G.neighbors(node))
            if current_comm.difference(best_comm) is not None:
                current_conductance=cal_conductance(G,current_comm)
                if current_conductance<best_conductance:
                    best_comm=current_comm
                    best_conductance=current_conductance
        selected_node+=1
        nodes=nodes-best_comm
        for node in best_comm:
            G.remove_node(node)
        end_flag+=len(best_comm)
        comm_temp=list(best_comm)
        comm_list[center_node]=comm_temp
    return comm_list