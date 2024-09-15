from torch import Tensor
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from .AttnAggregation import AttnAggregation
from torch_geometric.nn.aggr import SumAggregation, SoftmaxAggregation


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
        
        self.lsc_agg = AttnAggregation()
        self.sum_agg = SumAggregation()

        self.reset_parameters()
    
    def forward(
        self,
        x,
        score,
        edge_index,
        add_self_loop: bool = True,
        *args, 
        **kwargs
    ):  
        # add self loop
        
        #if add_self_loop:
        #    edge_index, _ = add_self_loops(edge_index)
        
        x = self.propagate(edge_index, x=x, score=score, *args, **kwargs)
        return x


    def message(self, x_j: Tensor, score_j:Tensor) -> Tensor:
        return x_j, score_j
        #return x_j*score_j
    
    

    def aggregate(self, inputs: Tensor, index: Tensor, ptr: Tensor | None = None, dim_size: int | None = None) -> Tensor:
        x, score = inputs
        x = self.lsc_agg.forward(attn=score, x=x, index=index, ptr=ptr, dim_size=dim_size)
        
        #x = inputs
        #x = self.sum_agg(x, index=index, ptr=ptr, dim_size=dim_size)
        
        return x

    
    def reset_parameters(self):
        pass