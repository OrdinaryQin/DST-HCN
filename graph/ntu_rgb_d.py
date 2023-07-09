import sys
import numpy as np
from . import hypergraph_utils as hgut
sys.path.extend(['../'])
from graph import tools

num_node = 25
self_link = [(i, i) for i in range(num_node)]
inward_ori_index = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
                    (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
                    (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
                    (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)]
inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
#neighbor = inward + outward
neighbor = inward + outward+self_link
num_node_1 = 11
indices_1 = [0, 3, 5, 7, 9, 11, 13, 15, 17, 19, 20]
self_link_1 = [(i, i) for i in range(num_node_1)]
inward_ori_index_1 = [(1, 11), (2, 11), (3, 11), (4, 3), (5, 11), (6, 5), (7, 1), (8, 7), (9, 1), (10, 9)]
inward_1 = [(i - 1, j - 1) for (i, j) in inward_ori_index_1]
outward_1 = [(j, i) for (i, j) in inward_1]
neighbor_1 = inward_1 + outward_1

num_node_2 = 5
indices_2 = [3, 5, 6, 8, 10]
self_link_2 = [(i ,i) for i in range(num_node_2)]
inward_ori_index_2 = [(0, 4), (1, 4), (2, 4), (3, 4), (0, 1), (2, 3)]
inward_2 = [(i - 1, j - 1) for (i, j) in inward_ori_index_2]
outward_2 = [(j, i) for (i, j) in inward_2]
neighbor_2 = inward_2 + outward_2

class Graph:
    def __init__(self, labeling_mode='spatial', scale=1):
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor
        self.A = self.get_adjacency_matrix(labeling_mode)
        self.A1 = tools.get_spatial_graph(num_node_1, self_link_1, inward_1, outward_1)
        self.A2 = tools.get_spatial_graph(num_node_2, self_link_2, inward_2, outward_2)
        self.A_binary = tools.edge2mat(neighbor, num_node)
        self.A_norm = tools.normalize_adjacency_matrix(self.A_binary + 2*np.eye(num_node))
        self.A_binary_K = tools.get_k_scale_graph(scale, self.A_binary)

        self.A_A1 = ((self.A_binary + np.eye(num_node)) / np.sum(self.A_binary + np.eye(self.A_binary.shape[0]), axis=1, keepdims=True))[indices_1]
        self.A1_A2 = tools.edge2mat(neighbor_1, num_node_1) + np.eye(num_node_1)
        self.A1_A2 = (self.A1_A2 / np.sum(self.A1_A2, axis=1, keepdims=True))[indices_2]


    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            X = tools.edge2mat(neighbor, num_node)
            A2 = tools.get_spatial_graph(num_node, self_link, inward, outward)[2]+tools.get_spatial_graph(num_node, self_link, inward, outward)[1]
            A_binary = tools.edge2mat(neighbor, num_node)
          #  A5 = tools.normalize_adjacency_matrix(A_binary + 2*np.eye(num_node))
            A5=X
            #A1 = tools.gen_knn_hg(X, n_neighbors=1)
            #A2 = tools.gen_knn_hg(X, n_neighbors=2)
    
            A5 = hgut.generate_G_from_H(A5)
                    
            A3 = tools.gen_knn_hg(X, n_neighbors=3)
       #     A3 = tools.gen_clustering_hg(X, n_clusters=6)
           # A3 = hgut.generate_G_from_H(A3)
            
            A4 = tools.gen_clustering_hg(X, n_clusters=4)
            
        #    x=A4
          #  a=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
          #  x=np.c_[x,a]
          #  x=np.c_[x,a]
          #  x=np.c_[x,a]
           # x[:,13]=0
           # x[(7,14,15,21,22),5]=1
           # x[(11,18,19,23,24),6]=1
           # x[(3,21,23,15,19),4]=1
          #  A4=x
            
            A4 = hgut.generate_G_from_H(A4)
           # A1=np.expand_dims(A1,axis=0)
            A2=np.expand_dims(A2,axis=0)
            A3=np.expand_dims(A3,axis=0)
            A4=np.expand_dims(A4,axis=0)
            A5=np.expand_dims(A5,axis=0)
            A = np.concatenate((A4, A2, A3,A5))
        else:
            raise ValueError()
        return A
#if __name__ == '__main__':
