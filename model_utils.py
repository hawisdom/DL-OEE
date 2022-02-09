import torch
import torch.nn as nn
import torch.nn.functional as F

def mask_logits(target, mask):
    return target * mask + (1 - mask) * (-1e30)

class AttrAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, args, concat=True):
        super(AttrAttentionLayer, self).__init__()
        self.dropout = args.dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = args.alpha
        self.concat = concat
        self.gat_our = args.gat_our

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj,level,node_type):
        Wh = torch.matmul(h, self.W)  # (N,L,2D)
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))
        if self.gat_our:
            e = torch.mul(e,node_type)
            Wh_temp = Wh.reshape(Wh.shape[0] * Wh.shape[1], Wh.shape[2])
            level_temp = level.reshape(level.shape[0] * level.shape[1], 1)
            c = torch.mul(Wh_temp, level_temp)
            Wh = c.reshape(Wh.shape)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.relu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[1]  # number of nodes

        # Below, two matrices are created that contain embeddings in their rows in different orders.
        # (e stands for embedding)
        # These are the rows of the first matrix (Wh_repeated_in_chunks):
        # e1, e1, ..., e1,            e2, e2, ..., e2,            ..., eN, eN, ..., eN
        # '-------------' -> N times  '-------------' -> N times       '-------------' -> N times
        #
        # These are the rows of the second matrix (Wh_repeated_alternating):
        # e1, e2, ..., eN, e1, e2, ..., eN, ..., e1, e2, ..., eN
        # '----------------------------------------------------' -> N times
        #
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=1)
        Wh_repeated_alternating = Wh.repeat(1,N, 1)

        # The all_combination_matrix, created below, will look like this (|| denotes concatenation):
        # e1 || e1
        # e1 || e2
        # e1 || e3
        # ...
        # e1 || eN
        # e2 || e1
        # e2 || e2
        # e2 || e3
        # ...
        # e2 || eN
        # ...
        # eN || e1
        # eN || e2
        # eN || e3
        # ...
        # eN || eN
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=2)
        return all_combinations_matrix.view(Wh.size()[0], N,N, 2 * self.out_features)

class DepAttentionLayer(nn.Module):
    def __init__(self, in_dim = 300, hidden_dim = 64,concat=True):
        super(DepAttentionLayer, self).__init__()

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.a = nn.Parameter(torch.empty(size=(1, 2*hidden_dim)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.concat = concat

    def forward(self,feature, dep_tags_v, dmask):
        '''
        Q dep_tags_v          [N, L, D]
        mask dmask          [N, L]
        '''
        Q = self.fc1(dep_tags_v)
        Q = self.relu(Q)
        Q = self.fc2(Q)  # (N, L, 1)
        Q = Q.squeeze(2)
        Q = F.softmax(mask_logits(Q, dmask), dim=1)

        Q = Q.unsqueeze(2)  #(N, L, 1)
        out = torch.bmm(feature.transpose(1, 2), Q)  #(N, 2D, 1)
        out = torch.matmul(out,self.a) #(N, 2D, 2D)
        out = torch.bmm(feature,out) #(N, L, 2D)

        if (self.concat):
            return self.relu(out)
        else:
            return out

class POSAttentionLayer(nn.Module):
    def __init__(self, in_dim = 300, hidden_dim = 64,concat=True):
        super(POSAttentionLayer, self).__init__()

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.a = nn.Parameter(torch.empty(size=(1, 2*hidden_dim)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.concat = concat

    def forward(self,feature,pos_tags_v,dmask):
        '''
        Q pos_tags_v      [N, L, D]
        mask dmask        [N, L]
        '''
        Q = self.fc1(pos_tags_v)
        Q = self.relu(Q)
        Q = self.fc2(Q)  # (N, L, 1)
        Q = Q.squeeze(2)
        Q = F.softmax(mask_logits(Q, dmask), dim=1)

        Q = Q.unsqueeze(2)  # (N, L, 1)
        out = torch.bmm(feature.transpose(1, 2), Q)  # (N, 2D, 1)
        out = torch.matmul(out, self.a)  # (N, 2D, 2D)
        out = torch.bmm(feature, out)  # (N, L, 2D)

        if (self.concat):
            return self.relu(out)
        else:
            return out