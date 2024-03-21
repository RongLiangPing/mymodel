import torch
import torch.nn as nn
import torch.nn.functional as F
from models.node_layer import GraphAttentionLayer, SpGraphAttentionLayer


class node_GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(node_GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]#定义多个注意力头，储存在列表中
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
            #add_module是Module类的成员函数，输入参数为Module.add_module(name: str, module: Module)。功能为，为Module添加一个子module，对应名字为name
            #add_module()函数也可以在GAT.init(self)以外定义A的子模块

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)
        #输出层的输入张量的shape之所以为(nhid * nheads, nclass)是因为在forward函数中多个注意力机制在同一个节点上得到的多个不同特征被拼接成了一个长的特征

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)  #将多个注意力机制在同一个节点上得到的多个不同特征进行拼接形成一个长特征
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj)) #该行代码即完成原文中的式子（6）
        return F.log_softmax(x, dim=1)  #F.log_softmax在数学上等价于log(softmax(x))，但做这两个单独操作速度较慢，数值上也不稳定。这个函数使用另一种公式来正确计算输出和梯度


class SpGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Sparse version of GAT."""
        super(SpGAT, self).__init__()
        self.dropout = dropout

        self.attentions = [SpGraphAttentionLayer(nfeat, 
                                                 nhid, 
                                                 dropout=dropout, 
                                                 alpha=alpha, 
                                                 concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = SpGraphAttentionLayer(nhid * nheads, 
                                             nclass, 
                                             dropout=dropout, 
                                             alpha=alpha, 
                                             concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)
