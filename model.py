import torch
import torch.nn as nn
import torch.nn.functional as F
from model_utils import AttrAttentionLayer,DepAttentionLayer,POSAttentionLayer
torch.set_printoptions(profile="full")


class MC_HGAT(nn.Module):
    def __init__(self, args, dep_tag_num, pos_tag_num):
        super(MC_HGAT, self).__init__()
        self.args = args

        num_embeddings, embed_dim = args.word2vec_embedding.shape
        self.embed = nn.Embedding(num_embeddings, embed_dim)
        self.embed.weight = nn.Parameter(
            args.word2vec_embedding, requires_grad=False)

        self.dropout = nn.Dropout(args.dropout)

        self.bilstm = nn.LSTM(input_size=args.embedding_dim, hidden_size=args.hidden_size,
                              bidirectional=True, batch_first=True, num_layers=args.num_layers)

        self.gat_attr = [
            AttrAttentionLayer(2 * args.hidden_size, 2 * args.hidden_size, args, concat=True).to(args.device) for i
            in range(args.num_heads)]
        self.out_att = AttrAttentionLayer(2 * args.hidden_size, 2 * args.hidden_size, args,
                                          concat=False).to(args.device)

        self.gat_dep = [DepAttentionLayer(args.embedding_dim,args.hidden_size,concat=False).to(args.device) for i in range(args.num_heads)]
        self.gat_pos = [POSAttentionLayer(args.embedding_dim,args.hidden_size,concat=False).to(args.device) for i in range(args.num_heads)]

        self.dep_embed = nn.Embedding(dep_tag_num, args.embedding_dim)
        self.pos_embed = nn.Embedding(pos_tag_num, args.embedding_dim)

        last_hidden_size = args.hidden_size * 6

        layers = [
            nn.Linear(last_hidden_size, args.final_hidden_size), nn.ReLU()]
        for _ in range(args.num_mlps-1):
            layers += [nn.Linear(args.final_hidden_size,
                                 args.final_hidden_size), nn.ReLU()]
        self.fcs = nn.Sequential(*layers)
        self.fc_final = nn.Linear(args.final_hidden_size, args.num_classes)

    def forward(self, tokens_ids, pos_class, dep_ids,text_len, level,adj,adj_node_type):
        fmask = (torch.zeros_like(tokens_ids) != tokens_ids).float()  # (N，L)

        attr_feature = self.embed(tokens_ids)  # (N, L, D)
        attr_feature = self.dropout(attr_feature) # (N, L, D)

        attr_feature, _ = self.bilstm(attr_feature) # (N,L,2D)

        dep_feature = self.dep_embed(dep_ids) # (N, L, D)
        dep_out = [g(attr_feature,dep_feature,fmask).unsqueeze(1) for g in self.gat_dep]  #(N, 1, L, 2D) * num_heads
        dep_out = torch.cat(dep_out, dim = 1) #(N,num_heads,L,2D)
        dep_out = dep_out.mean(dim = 1) #(N,L,2D)

        pos_feature = self.pos_embed(pos_class) # (N, L, D)
        pos_out = [g(attr_feature,pos_feature,fmask).unsqueeze(1) for g in self.gat_pos]  # (N, 1, L, 2D) * num_heads
        pos_out = torch.cat(pos_out, dim=1)  #(N,num_heads,L,2D)
        pos_out = pos_out.mean(dim=1)  #(N,L,2D)

        attr_out = [g(attr_feature,adj, level, adj_node_type).unsqueeze(1) for g in
                    self.gat_attr]  # (N,1,L,2D) * num_heads
        attr_out = torch.cat(attr_out, dim=1)  # (N,num_heads,L,2D)
        attr_out = attr_out.mean(dim=1)  # (N,L,2D)
        attr_out = self.dropout(attr_out) # (N,L,2D)
        attr_out = F.relu(self.out_att(attr_out, adj, level, adj_node_type))  # (N,L,2D)

        all_feature_out = torch.cat([dep_out,pos_out,attr_out], dim = 2) # (N, L, 6D)
        
        out = self.dropout(all_feature_out)
        out = self.fcs(out)
        logit = self.fc_final(out)
        logit = logit.view(logit.shape[0] * logit.shape[1], logit.shape[2])
        return logit




