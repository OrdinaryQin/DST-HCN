import numpy as np
from sklearn.metrics import pairwise_distances
import scipy.sparse as sparse
from sklearn.cluster import KMeans
def get_sgp_mat(num_in, num_out, link):
    A = np.zeros((num_in, num_out))
    for i, j in link:
        A[i, j] = 1
    A_norm = A / np.sum(A, axis=0, keepdims=True)
    return A_norm

def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A

def get_k_scale_graph(scale, A):
    if scale == 1:
        return A
    An = np.zeros_like(A)
    A_power = np.eye(A.shape[0])
    for k in range(scale):
        A_power = A_power @ A
        An += A_power
    An[An > 0] = 1
    return An

def normalize_digraph(A):
    Dl = np.sum(A, 0)
    h, w = A.shape
    Dn = np.zeros((w, w))
    for i in range(w):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD


def get_spatial_graph(num_node, self_link, inward, outward):
    I = edge2mat(self_link, num_node)
    In = normalize_digraph(edge2mat(inward, num_node))
    Out = normalize_digraph(edge2mat(outward, num_node))
    A = np.stack((I, In, Out))
    return A

def normalize_adjacency_matrix(A):
    node_degrees = A.sum(-1)
    degs_inv_sqrt = np.power(node_degrees, -0.5)
    norm_degs_matrix = np.eye(len(node_degrees)) * degs_inv_sqrt
    return (norm_degs_matrix @ A @ norm_degs_matrix).astype(np.float32)


def k_adjacency(A, k, with_self=False, self_factor=1):
    assert isinstance(A, np.ndarray)
    I = np.eye(len(A), dtype=A.dtype)
    if k == 0:
        return I
    Ak = np.minimum(np.linalg.matrix_power(A + I, k), 1) \
       - np.minimum(np.linalg.matrix_power(A + I, k - 1), 1)
    if with_self:
        Ak += (self_factor * I)
    return Ak

def get_multiscale_spatial_graph(num_node, self_link, inward, outward):
    I = edge2mat(self_link, num_node)
    A1 = edge2mat(inward, num_node)
    A2 = edge2mat(outward, num_node)
    A3 = k_adjacency(A1, 2)
    A4 = k_adjacency(A2, 2)
    A1 = normalize_digraph(A1)
    A2 = normalize_digraph(A2)
    A3 = normalize_digraph(A3)
    A4 = normalize_digraph(A4)
    A = np.stack((I, A1, A2, A3, A4))
    return A



def get_uniform_graph(num_node, self_link, neighbor):
    A = normalize_digraph(edge2mat(neighbor + self_link, num_node))
    return A

def gen_knn_hg(X, n_neighbors, is_prob=True, with_feature=False):
    n_nodes = X.shape[0]
    n_edges = n_nodes
    
    m_dist = pairwise_distances(X)#计算欧式距离，还有其它方法
    
    # top n_neighbors+1
    m_neighbors = np.argpartition(m_dist, kth=n_neighbors+1, axis=1)#每个节点找出最小前五个点索引
    m_neighbors_val = np.take_along_axis(m_dist, m_neighbors, axis=1)#根据索引返回前五个最小距离数据值
    
    m_neighbors = m_neighbors[:, :n_neighbors+1]#每个节点找出最小前五个点索引
    m_neighbors_val = m_neighbors_val[:, :n_neighbors+1]
    
    # check
    for i in range(n_nodes):
        if not np.any(m_neighbors[i, :] == i):
            m_neighbors[i, -1] = i
            m_neighbors_val[i, -1] = 0.
    
    node_idx = m_neighbors.reshape(-1)#展平点索引
    edge_idx = np.tile(np.arange(n_edges).reshape(-1, 1), (1, n_neighbors+1)).reshape(-1)#构造边索引
    
    if not is_prob:
        values = np.ones(node_idx.shape[0])
    else:
        avg_dist = np.mean(m_dist)
        m_neighbors_val = m_neighbors_val.reshape(-1)
        values = np.exp(-np.power(m_neighbors_val, 2.) / np.power(avg_dist, 2.))#权重公式
    
    H = sparse.coo_matrix((values, (node_idx, edge_idx)), shape=(n_nodes, n_edges))
    A=H.todense()
    return A
def gen_clustering_hg(X, n_clusters, method="kmeans", with_feature=False, random_state=None):
    """
    :param X: numpy array, shape = (n_samples, n_features)
    :param n_clusters: int, number of clusters
    :param method: str, clustering methods("kmeans",)
    :param with_feature: bool, optional(default=False)
    :param random_state: int, optional(default=False) determines random number generation
    for centroid initialization
    :return: instance of HyperG
    """
    if method == "kmeans":
        cluster = KMeans(n_clusters=n_clusters, random_state=random_state).fit(X).labels_
    else:
        raise ValueError("{} method is not supported".format(method))

    assert n_clusters >= 1

    n_edges = n_clusters
    n_nodes = X.shape[0]

    node_idx = np.arange(n_nodes)
    edge_idx = cluster

    values = np.ones(node_idx.shape[0])
    H = sparse.coo_matrix((values, (node_idx, edge_idx)), shape=(n_nodes, n_edges))
    A=H.todense()
    return A
    