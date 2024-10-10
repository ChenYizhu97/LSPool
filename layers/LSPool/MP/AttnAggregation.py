
from typing import Optional

import torch
from torch import Tensor

from torch_geometric.nn.aggr import Aggregation
from torch_geometric.utils import softmax
from torch.nn import Parameter

class AttnAggregation(Aggregation):
    r"""An aggregation operator that takes the feature-wise maximum across a
    set of elements

    .. math::
        \mathrm{max}(\mathcal{X}) = \max_{\mathbf{x}_i \in \mathcal{X}}
        \mathbf{x}_i.
    """
    def __init__(self, t: float = 1.0, learn: bool = False):
        super().__init__()
        self._init_t = t
        self.learn = learn
        self.t = Parameter(torch.empty(1)) if learn else t
        self.reset_parameters()
        
    def forward(self, attn: Tensor, x:Tensor, index: Optional[Tensor] = None,
                ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
                dim: int = -2) -> Tensor:
        
        t = self.t
        
        if not isinstance(t, (int, float)) or t != 1:
            # multiply attn by temperature if set temperature learnable
            attn = attn * t

        if not self.learn:
            # softmax without temperature 
            with torch.no_grad():
                attn = softmax(attn, index, ptr, dim_size, dim)
        else:
            # softmax with learnable temperature
            attn = softmax(attn, index, ptr, dim_size, dim)
        return self.reduce(x * attn, index, ptr, dim_size, dim, reduce='sum')

    def reset_parameters(self):
        if isinstance(self.t, Tensor):
            self.t.data.fill_(self._init_t)

