import numpy as np
import scipy.sparse as sp
import torch
import networkx as nx
from node2vec import Node2Vec

'''
def encode_onehot(labels):
    # The classes must be sorted before encoding to enable static class encoding.
    # In other words, make sure the first class always maps to index 0.
    # 构建标签的one-hot编码
    # 特征矩阵features每一行的特征对应于一个节点，第一行特征对应的节点的编号并不是0，而是cora数据集采集数据的时候已经固定了的。
    # 在这里，我们将features第一行的特征所对应的节点的标签设置为[1,0,0,0,0,...,0]，以此类推，第二行的特征所对应的节点的标签设置为[0,1,0,0,0,...,0]
    classes = sorted(list(set(labels)))
    #这些标签是文本标签而不是数值标签
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    #构造字典，key是标签（文本标签），value是对应的one-hot形式的标签
    #np.identity创建对角为1的方阵，i的取值分别是0~len(classes)
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    #map函数是python的内置函数，第一个参数是一个function，该方法会根据第二个参数的每一个元素调用 function 函数，最终返回包含每次 function 函数返回值的新列表
    return labels_onehot
'''

def load_data_with_node2vec(path="./data/cora/", dataset="cora"):
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))#加载数据
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)#创建系数特征矩阵

    # 构建图数据结构
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    graph = nx.from_edgelist(edges)

    # 使用 Node2Vec 方法生成节点嵌入向量
    node2vec = Node2Vec(graph, dimensions=64, walk_length=30, num_walks=200, workers=4)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)

    # 将节点嵌入向量作为标签
    labels = np.array([model[str(node)] for node in graph.nodes()])

    # 构建邻接矩阵等
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    features = normalize_features(features)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    adj = torch.FloatTensor(np.array(adj.todense()))
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.FloatTensor(labels)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))    #求每一行的和
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()   #D^{-0.5}
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)   #D^{-0.5}
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)
    # mx.dot(r_mat_inv_sqrt).transpose()是(AD^{-0.5})^T
    # mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)是(AD^{-0.5})^T·D^{-0.5}=D^{-0.5}AD^{-0.5}


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))    
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)    #max(1)返回每一行中最大值的那个元素所构成的一维张量，且返回对应的一维索引张量（返回最大元素在这一行的列索引）
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)