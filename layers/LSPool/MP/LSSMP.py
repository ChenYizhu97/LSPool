from typing import Optional, Union
import torch
from torch import Tensor
from torch.nn import Linear
from torch_geometric.nn import MessagePassing, SumAggregation
from torch_geometric.nn.resolver import activation_resolver
from .AttnAggregation import AttnAggregation

eps=1e-7

class LSSMP(MessagePassing):
    """Local set score MP that calculates score of local set.

    Args:
        x: Node features.
        edge_index: Sparse adjacency matrix.
        edge_attr: Edge attributes (optional).
    Returns:
    
    """
    def __init__(
            self, 
            in_channels: int,
            in_channels_edge: Optional[int] = None,
            nonlinearity: str = "elu",
            act: Optional[str] = "tanh",
            k: Optional[int] = None,
            *args, 
            **kwargs
        ):
        super().__init__(aggr=None, *args, **kwargs)

        hidden_channels = 128 
        #if k (clusters number) is defined, the output of this layer is the assignment matrix rather than scores.
        out_dim = 1 if k is None else k

        self.L_q = Linear(in_channels, hidden_channels)
        self.L_k = Linear(in_channels, hidden_channels)

        self.L_d = Linear(in_channels, hidden_channels)
        self.L_fd = Linear(hidden_channels, hidden_channels)
        
        self.L_sx = Linear(hidden_channels, 1)
        self.L_sd = Linear(hidden_channels, 1)

        #nonlinearity
        self.nonlinearity = activation_resolver(nonlinearity)

        self.sum_agg = SumAggregation()
        self.attn_agg = AttnAggregation()
        if in_channels_edge is not None:
            self.L_fe = Linear(in_channels, hidden_channels)
            self.L_se = Linear(hidden_channels, 1)
            self.L_s = Linear(3, out_dim)
        else:
            self.L_fe = None
            self.L_se = None
            self.L_s = Linear(2, out_dim)

        #tanh?
        self.act = activation_resolver(act) if act is not None else None
        self.reset_parameters()

    
    def forward(
            self, 
            x: Tensor,
            edge_index: Tensor,
            edge_attr: Optional[Tensor],
            *args, 
            **kwargs
        ):

        scores = self.propagate(edge_index=edge_index, edge_attr=edge_attr, x=x, *args, **kwargs)

        return scores
        


    def message(self, x_j: Tensor, x_i: Tensor, edge_attr_j:Optional[Tensor]) -> Tensor:
        diff = self.nonlinearity(self.L_d(x_j)) - self.nonlinearity(self.L_d(x_i))
        feat_diff = self.nonlinearity(self.L_fd(diff))

        q_x_i = self.nonlinearity(self.L_q(x_i))
        k_x_j = self.nonlinearity(self.L_k(x_j))
        similarity = (q_x_i * k_x_j).sum(dim = -1).unsqueeze(-1)

        if edge_attr_j is not None:
            feat_e = self.nonlinearity(self.L_fe(edge_attr_j))
        else:
            feat_e = None

        return similarity, x_j, feat_diff, feat_e
    
    def aggregate(self, inputs: Tensor, index: Tensor, ptr: Tensor | None = None, dim_size: int | None = None) -> Tensor:

        similarity, x_j, feat_diff, feat_e = inputs

        feat_x = self.attn_agg.forward(similarity, x_j, index=index, ptr=ptr, dim_size=dim_size)
        feat_diff = self.sum_agg.forward(feat_diff, index=index, ptr=ptr, dim_size=dim_size)
        #feat_diff = self.attn_agg.forward(similarity, feat_diff, index=index, ptr=ptr, dim_size=dim_size)
        if feat_e is not None:
            feat_e = self.sum_agg.forward(feat_e, index=index, ptr=ptr, dim_size=dim_size)
            #feat_e = self.attn_agg.forward(similarity, feat_e, index=index, ptr=ptr, dim_size=dim_size)
            
        return feat_x, feat_diff, feat_e
    
    def update(self, inputs: Tensor) -> Tensor:
        feat_x, feat_diff, feat_e = inputs

        #score for each feature
        s_x = self.nonlinearity(self.L_sx(feat_x))
        s_d = self.nonlinearity(self.L_sd(feat_diff))

        if feat_e is not None:
            s_e = self.nonlinearity(self.L_se(feat_e))
            s = torch.concatenate((s_x, s_d, s_e), -1)
        else:
            s = torch.concatenate((s_x, s_d), -1)

        scores = self.L_s(s)
        if self.act is not None:
            scores = self.act(scores)

        return scores

    def reset_parameters(self):
        self.L_d.reset_parameters()
        self.L_k.reset_parameters()
        self.L_q.reset_parameters()
        self.L_fd.reset_parameters()
        if self.L_fe is not None: self.L_fe.reset_parameters()
        self.L_sd.reset_parameters()
        self.L_sx.reset_parameters()
        if self.L_se is not None: self.L_se.reset_parameters()
        self.L_s.reset_parameters()
        
        