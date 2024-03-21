import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout #dropout率
        self.in_features = in_features #输入特征维度
        self.out_features = out_features #输出特征维度
        self.alpha = alpha #激活函数的参数
        self.concat = concat #是否使用拼接的方法来组合多注意力头的结果

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))    #创建一个大小为（in_features,out_features)的张量作为权重矩阵W
        nn.init.xavier_uniform_(self.W.data, gain=1.414)    #对权重矩阵进行初始化
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))    #a是shape为(out_features',1)的张量，为注意力参数
        nn.init.xavier_uniform_(self.a.data, gain=1.414)    #初始化a

        self.leakyrelu = nn.LeakyReLU(self.alpha)  #建立relu函数实例

    def forward(self, h, adj, deg):
        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)，节点特征与w相乘，得到新的特征表示Wh
        a_input = self.node_prepare_attentional_mechanism_input(Wh) #特征拼接操作 Wh_i||Wh_j ，得到一个shape = (N ， N, 2 * out_features)的新特征矩阵
        
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))#aT * Wh_i
        #torch.matmul(a_input, self.a)的shape=(N,N,1)，经过squeeze(2)后，shape变为(N,N)
        
        #我可以直接计算一个邻接矩阵，其中第i行第j列就表示dvi/dvi+dvj
        zero_vec = -9e15*torch.ones_like(e)

        # 计算参数矩阵，只在邻接矩阵中大于零的位置进行乘法操作
        #deg为一个N*N的矩阵，其中第i行第j列为d_vi/(d_vi+d_vj)
        param_matrix = deg * (adj > 0).float()

        # 对参数矩阵与attention相乘，只在邻接矩阵中大于零的位置进行乘法操作
        attention = torch.where(adj > 0, e * param_matrix, torch.zeros_like(e))
 
        attention = F.softmax(attention, dim=1)  #对每一行内的数据做归一化，就是除以分母
        
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)   #当输入是都是二维时，就是普通的矩阵乘法，和tensor.mm函数用法相同
        #h_prime.shape=(N,out_features)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def node_prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0] # 获取节点特征矩阵 Wh 的大小，其中 N 是节点的数量。
        
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        #repeat方法可以对 Wh 张量中的单维度和非单维度进行复制操作，并且会真正的复制数据保存到内存中
        #repeat(N, 1)表示dim=0维度的数据复制N份，dim=1维度的数据保持不变

        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        # all_combinations_matrix.shape == (N * N, 2 * out_features)

        return all_combinations_matrix.view(N, N, 2 * self.out_features)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class SpecialSpmmFunction(torch.autograd.Function): #虽然pytorch可以自动求导，但是有时候一些操作是不可导的，这时候你需要自定义求导方式。也就是所谓的 “Extending torch.autograd”
    # 创建torch.autograd.Function类的一个子类
    # 必须是staticmethod
    @staticmethod
    def forward(ctx, indices, values, shape, b):    #forward()可以有任意多个输入、任意多个输出
    # 第一个是ctx，第二个是input，其他是可选参数。
    # ctx在这里类似self，ctx的属性可以在backward中调用。
    # 自己定义的Function中的forward()方法，所有的Variable参数将会转成tensor！因此这里的input也是tensor．在传入forward前，autograd engine会自动将Variable unpack成Tensor。
        assert indices.requires_grad == False   #检查条件，不符合就终止程序
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b) # 将Tensor转变为Variable保存到ctx中，在backward中需要这些变量
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output): #backward()的输入和输出的个数就是forward()函数的输出和输入的个数。其中，backward()的输入表示关于forward()的输出的梯度(计算图中上一节点的梯度)，
    #backward()的输出表示关于forward()的输入的梯度。在输入不需要梯度时（通过查看needs_input_grad参数）或者不可导时，可以返回None。
        a, b = ctx.saved_tensors    
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]: # 判断forward()输入的Variable（不包括上下文ctx）是否需要进行反向求导计算梯度
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b) #调用SpecialSpmmFunctionde1的forward()

    
class SpGraphAttentionLayer(nn.Module):

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)
                
        self.a = nn.Parameter(torch.zeros(size=(1, 2*out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def forward(self, input, adj):
        dv = 'cuda' if input.is_cuda else 'cpu'

        N = input.size()[0]
        edge = adj.nonzero().t()
        # edge：2 x E（E是边的数量）

        h = torch.mm(input, self.W)
        # h: N x out
        assert not torch.isnan(h).any()

        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        # edge: 2*D x E

        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))   
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N,1), device=dv))   #special_spmm是我们定制的稀疏的矩阵乘法函数
        # e_rowsum: N x 1

        edge_e = self.dropout(edge_e)
        # edge_e: E

        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out
        
        h_prime = h_prime.div(e_rowsum) #除以每一行的“行和”，从而正则化
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()

        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'