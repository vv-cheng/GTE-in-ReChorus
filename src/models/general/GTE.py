import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
import time

from models.BaseModel import GeneralModel

class GTE(GeneralModel):
    reader = 'BaseReader'
    runner = 'GTERunner'
    extra_log_args = ['k']
    
    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--k', type=int, default=3)
        parser.add_argument('--normalize', type=int, default=0,
                           help='Whether to normalize during propagation.')
        return GeneralModel.parse_model_args(parser)
    
    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        
        self.k = args.k
        self.normalize = args.normalize
        self.propagation_done = False
        self.user_rep_sparse = None
        self.corpus = corpus
        self.device = args.device
        
        print(f"[GTE] Initialized: k={self.k}, users={self.user_num}, items={self.item_num}")
    
    def _ensure_propagation(self):
        if not self.propagation_done:
            self._perform_propagation()
    
    def _perform_propagation(self):
        start_time = time.time()
        print(f"[GTE] Starting propagation (k={self.k}, normalize={self.normalize})...")
        
        train_df = self.corpus.data_df['train']
        rows = train_df['user_id'].values.astype(np.int32)
        cols = train_df['item_id'].values.astype(np.int32)
        
        print(f"  Interactions: {len(rows)}")
        print(f"  Raw density: {len(rows)/(self.user_num*self.item_num):.6f}")
        
        adj = sp.coo_matrix((np.ones(len(rows)), (rows, cols)),
                           shape=(self.user_num, self.item_num))
        
        if self.normalize:
            row_sum = np.array(adj.sum(axis=1)).flatten()
            row_sum[row_sum == 0] = 1
            diag = sp.diags(1.0 / row_sum)
            adj = diag.dot(adj)
        
        adj_csr = adj.tocsr()
        
        H_i = sp.eye(self.item_num, format='csr')
        H_u = sp.csr_matrix((self.user_num, self.item_num))
        
        for i in range(self.k):
            H_u_new = adj_csr.dot(H_i) + H_u
            H_i_new = adj_csr.T.dot(H_u) + H_i
            
            if self.normalize and i % 2 == 0:
                H_u_new = self._normalize(H_u_new)
                H_i_new = self._normalize(H_i_new)
            
            H_u, H_i = H_u_new.tocsr(), H_i_new.tocsr()
            
            density = H_u.nnz / (self.user_num * self.item_num)
            print(f"  Layer {i+1}: density={density:.6f}, nnz={H_u.nnz}")
        
        self.user_rep_sparse = H_u
        self.propagation_done = True
        
        print(f"[GTE] Propagation done in {time.time()-start_time:.2f}s")
    
    def _normalize(self, mat):
        row_sum = np.array(mat.sum(axis=1)).flatten()
        row_sum[row_sum == 0] = 1
        diag = sp.diags(1.0 / row_sum)
        return diag.dot(mat)
    
    def forward_batch(self, user_ids, item_ids):
        self._ensure_propagation()
        
        unique_users = np.unique(user_ids)
        user_reps = self.user_rep_sparse[unique_users].toarray()

        user_to_idx = {uid: i for i, uid in enumerate(unique_users)}
        
        predictions = np.zeros((len(user_ids), item_ids.shape[1]))
        
        for i in range(len(user_ids)):
            uid = user_ids[i]
            if uid in user_to_idx:
                rep = user_reps[user_to_idx[uid]]
                predictions[i] = rep[item_ids[i]]
        
        return torch.FloatTensor(predictions)
    
    def forward(self, feed_dict):
        user_ids = feed_dict['user_id'].long().cpu().numpy()
        item_ids = feed_dict['item_id'].long().cpu().numpy()
        
        if user_ids.ndim > 1:
            user_ids = user_ids.squeeze(1)
        
        predictions = self.forward_batch(user_ids, item_ids)
        return {'prediction': predictions.to(self.device)}
    
    def loss(self, out_dict):
        return torch.tensor(0.0)
    
    def customize_parameters(self):
        return []