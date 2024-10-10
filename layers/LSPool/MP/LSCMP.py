from torch import Tensor
from torch_geometric.nn import MessagePassing
from .AttnAggregation import AttnAggregation
from torch_geometric.nn.aggr import SumAggregation


class LSCMP(MessagePassing):
    """Local set collapse MP that collapses local sets into pooled nodes.

    Args:
        scores: Scores produced by LSSMP.
        x: Node features.
        edge_index: Sparse adjancency matrix.
    """
    def __init__(
        self,
        *args,
        **kwargs
    ):
        super().__init__(aggr=None, *args, **kwargs)
        
        self.attn_agg = AttnAggregation(learn=True)
        #self.sum_agg = SumAggregation()
        #self.lattn_agg = LearnableAttnAggregation(learn=True)
        self.reset_parameters()
    
    def forward(
        self,
        x,
        score,
        edge_index,
        *args, 
        **kwargs
    ):  
        
        x = self.propagate(edge_index, x=x, score=score, *args, **kwargs)
        return x


    def message(self, x_j: Tensor, score_j:Tensor) -> Tensor:
        #return x_j, score_j
        return x_j*score_j
    
     

    def aggregate(self, inputs: Tensor, index: Tensor, ptr: Tensor | None = None, dim_size: int | None = None) -> Tensor:
        #x, score = inputs
        #x = self.attn_agg.forward(attn=score, x=x, index=index, ptr=ptr, dim_size=dim_size)
        x = inputs
        x = self.sum_agg(x, index=index, ptr=ptr, dim_size=dim_size)
        
        return x

    
    def reset_parameters(self):
        # self.attn_agg.reset_parameters()
        pass