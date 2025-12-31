import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp

from models.BaseModel import GeneralModel


class GCCF(GeneralModel):
    reader = 'BaseReader'
    runner = 'BaseRunner'
    extra_log_args = ['emb_size', 'n_layers', 'use_layer_norm']
    
    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64,
                            help='Size of embedding vectors.')
        parser.add_argument('--n_layers', type=int, default=2,
                            help='Number of GCN layers.')
        parser.add_argument('--use_layer_norm', type=int, default=1,
                            help='Whether to use layer normalization.')
        parser.add_argument('--activation', type=str, default='relu',
                            choices=['relu', 'leaky_relu', 'elu', 'none'],
                            help='Activation function type.')
        parser.add_argument('--init_scale', type=float, default=0.1,
                            help='Initialization scale for embeddings.')
        return GeneralModel.parse_model_args(parser)
    
    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self.emb_size = args.emb_size
        self.n_layers = args.n_layers
        self.use_layer_norm = args.use_layer_norm
        self.activation = args.activation
        self.init_scale = args.init_scale
        
        self.corpus = corpus
        
        self.norm_adj_mat = self._build_adj_matrix()
        
        self._define_params()
        self.apply(self.init_weights)

        self.check_list = []
    
    def _build_adj_matrix(self):
        user_count, item_count = self.user_num, self.item_num
        
        train_mat = self.corpus.train_clicked_set
        
        R = sp.dok_matrix((user_count, item_count), dtype=np.float32)
        for user in train_mat:
            for item in train_mat[user]:
                R[user, item] = 1
        R = R.tolil()
        
        adj_mat = sp.dok_matrix((user_count + item_count, user_count + item_count), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        
        adj_mat[:user_count, user_count:] = R
        adj_mat[user_count:, :user_count] = R.T
        adj_mat = adj_mat.todok()
        
        rowsum = np.array(adj_mat.sum(1)) + 1e-10
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        
        norm_adj_mat = d_mat_inv_sqrt.dot(adj_mat).dot(d_mat_inv_sqrt)
        return norm_adj_mat.tocoo()
    
    def _define_params(self):
        self.user_emb = nn.Embedding(self.user_num, self.emb_size)
        self.item_emb = nn.Embedding(self.item_num, self.emb_size)
        
        nn.init.normal_(self.user_emb.weight, mean=0.0, std=self.init_scale)
        nn.init.normal_(self.item_emb.weight, mean=0.0, std=self.init_scale)
        
        self.gcn_layers = nn.ModuleList()
        
        if self.use_layer_norm:
            self.layer_norms = nn.ModuleList()
        
        for layer in range(self.n_layers):
            linear_layer = nn.Linear(self.emb_size, self.emb_size, bias=True)
            nn.init.xavier_uniform_(linear_layer.weight, gain=1.0)
            nn.init.zeros_(linear_layer.bias)
            self.gcn_layers.append(linear_layer)
            
            if self.use_layer_norm:
                self.layer_norms.append(nn.LayerNorm(self.emb_size))
    
    def _apply_activation(self, x):
        if self.activation == 'none':
            return x
        elif self.activation == 'relu':
            return F.relu(x)
        elif self.activation == 'leaky_relu':
            return F.leaky_relu(x, negative_slope=0.1)
        elif self.activation == 'elu':
            return F.elu(x)
        else:
            return x
    
    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        rows = coo.row.astype(np.int64)
        cols = coo.col.astype(np.int64)
        indices = np.vstack([rows, cols])
        indices = torch.from_numpy(indices).long()
        values = torch.from_numpy(coo.data.astype(np.float32))
        return torch.sparse.FloatTensor(indices, values, coo.shape)
    
    def forward(self, feed_dict):
        users = feed_dict['user_id']
        items = feed_dict['item_id']
        
        if users.dtype != torch.long:
            users = users.long()
        if items.dtype != torch.long:
            items = items.long()
        
        if not hasattr(self, 'sparse_norm_adj'):
            self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(self.norm_adj_mat)
            self.sparse_norm_adj = self.sparse_norm_adj.coalesce().to(self.device)
        
        user_emb0 = self.user_emb.weight
        item_emb0 = self.item_emb.weight
        ego_embeddings = torch.cat([user_emb0, item_emb0], 0)
        
        all_embeddings = [ego_embeddings]
        current_embeddings = ego_embeddings
        
        for layer in range(self.n_layers):
            propagated = torch.sparse.mm(self.sparse_norm_adj, current_embeddings)
            
            transformed = self.gcn_layers[layer](propagated)
            
            if self.use_layer_norm:
                transformed = self.layer_norms[layer](transformed)
            
            activated = self._apply_activation(transformed)
            
            current_embeddings = current_embeddings + activated
            
            if self.dropout > 0 and self.training:
                current_embeddings = F.dropout(current_embeddings, p=self.dropout, training=self.training)
            
            all_embeddings.append(current_embeddings)
        
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        
        user_all_embeddings = all_embeddings[:self.user_num, :]
        item_all_embeddings = all_embeddings[self.user_num:, :]
        
        user_embeddings = user_all_embeddings[users, :]
        
        if len(items.shape) == 1:
            item_embeddings = item_all_embeddings[items, :]
            prediction = (user_embeddings * item_embeddings).sum(dim=1)
        else:
            batch_size, n_candidates = items.shape
            item_embeddings = item_all_embeddings[items.view(-1), :]
            item_embeddings = item_embeddings.view(batch_size, n_candidates, -1)
            
            user_embeddings_expanded = user_embeddings.unsqueeze(1)
            
            prediction = (user_embeddings_expanded * item_embeddings).sum(dim=2)
        
        return {'prediction': prediction}
    
    def loss(self, out_dict):
        return super().loss(out_dict)