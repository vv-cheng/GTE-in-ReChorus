import torch
import torch.nn as nn
import torch.nn.functional as F

from models.BaseModel import GeneralModel
from models.BaseImpressionModel import ImpressionModel

class SimGRACEBase(object):
    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64,
                            help='Size of embedding vectors.')
        parser.add_argument('--tau', type=float, default=0.5,
                            help='Temperature parameter for contrastive loss.')
        parser.add_argument('--contrastive_weight', type=float, default=0.1,
                            help='Weight for contrastive loss.')
        parser.add_argument('--dropout_rate', type=float, default=0.1,
                            help='Dropout rate for embedding perturbation.')
        return parser
    
    def _base_init(self, args, corpus):
        self.emb_size = args.emb_size
        self.tau = args.tau
        self.contrastive_weight = args.contrastive_weight
        self.dropout_rate = args.dropout_rate
        self._base_define_params()
        self.apply(self.init_weights)
    
    def _base_define_params(self):
        self.u_embeddings = nn.Embedding(self.user_num, self.emb_size)
        self.i_embeddings = nn.Embedding(self.item_num, self.emb_size)
        
        self.embedding_dropout = nn.Dropout(self.dropout_rate)
    
    def forward(self, feed_dict):
        u_ids = feed_dict['user_id']
        i_ids = feed_dict['item_id']

        if u_ids.dtype != torch.long:
            u_ids = u_ids.long()
        if i_ids.dtype != torch.long:
            i_ids = i_ids.long()

        cf_u_vectors = self.u_embeddings(u_ids)
        cf_i_vectors = self.i_embeddings(i_ids)

        prediction = (cf_u_vectors[:, None, :] * cf_i_vectors).sum(dim=-1)

        out_dict = {
            'prediction': prediction.view(feed_dict['batch_size'], -1),
            'u_v': cf_u_vectors,
            'i_v': cf_i_vectors
        }
        
        return out_dict

class SimGRACE(GeneralModel, SimGRACEBase):
    reader = 'BaseReader'
    runner = 'BaseRunner'
    extra_log_args = ['emb_size', 'tau', 'contrastive_weight', 'dropout_rate', 'batch_size']
    
    @staticmethod
    def parse_model_args(parser):
        parser = SimGRACEBase.parse_model_args(parser)
        return GeneralModel.parse_model_args(parser)
    
    def __init__(self, args, corpus):
        GeneralModel.__init__(self, args, corpus)
        self._base_init(args, corpus)
        self.check_list = []
    
    def loss(self, out_dict: dict) -> torch.Tensor:
        """
        Combined loss: BPR loss + contrastive loss
        """
        predictions = out_dict['prediction']
        pos_pred, neg_pred = predictions[:, 0], predictions[:, 1:]
        neg_softmax = (neg_pred - neg_pred.max()).softmax(dim=1)
        bpr_loss = -(((pos_pred[:, None] - neg_pred).sigmoid() * neg_softmax).sum(dim=1)).clamp(min=1e-8, max=1-1e-8).log().mean()
        
        u_vectors = out_dict['u_v']
        i_vectors = out_dict['i_v']

        u_vectors_view1 = self.embedding_dropout(u_vectors.clone())
        u_vectors_view2 = self.embedding_dropout(u_vectors.clone())

        pos_i_vectors = i_vectors[:, 0, :]
        pos_i_vectors_view1 = self.embedding_dropout(pos_i_vectors.clone())
        pos_i_vectors_view2 = self.embedding_dropout(pos_i_vectors.clone())

        def _compute_contrastive_loss(z1, z2):
            z1 = F.normalize(z1, dim=-1)
            z2 = F.normalize(z2, dim=-1)
            

            similarity_matrix = torch.matmul(z1, z2.T) / self.tau
            
            pos_sim = torch.diag(similarity_matrix)

            numerator = torch.exp(pos_sim)
            denominator = torch.exp(similarity_matrix).sum(dim=1) - torch.exp(torch.diag(similarity_matrix))
            
            contrastive_loss = -torch.log(numerator / denominator).mean()
            return contrastive_loss

        user_contrastive_loss = _compute_contrastive_loss(u_vectors_view1, u_vectors_view2)

        item_contrastive_loss = _compute_contrastive_loss(pos_i_vectors_view1, pos_i_vectors_view2)

        contrastive_loss = (user_contrastive_loss + item_contrastive_loss) / 2

        total_loss = bpr_loss + self.contrastive_weight * contrastive_loss
        
        return total_loss
    
    def forward(self, feed_dict):
        return SimGRACEBase.forward(self, feed_dict)

class SimGRACEImpression(ImpressionModel, SimGRACEBase):
    reader = 'ImpressionReader'
    runner = 'ImpressionRunner'
    extra_log_args = ['emb_size', 'tau', 'contrastive_weight', 'dropout_rate', 'batch_size']
    
    @staticmethod
    def parse_model_args(parser):
        parser = SimGRACEBase.parse_model_args(parser)
        return ImpressionModel.parse_model_args(parser)
    
    def __init__(self, args, corpus):
        ImpressionModel.__init__(self, args, corpus)
        self._base_init(args, corpus)
        self.check_list = []
    
    def forward(self, feed_dict):
        return SimGRACEBase.forward(self, feed_dict)