
from typing import Optional

import torch
from torch import Tensor

from torch_geometric.nn.aggr import Aggregation
from torch_geometric.utils import softmax

class AttnAggregation(Aggregation):
    r"""An aggregation operator that takes the feature-wise maximum across a
    set of elements

    .. math::
        \mathrm{max}(\mathcal{X}) = \max_{\mathbf{x}_i \in \mathcal{X}}
        \mathbf{x}_i.
    """
    def forward(self, attn: Tensor, x:Tensor, index: Optional[Tensor] = None,
                ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
                dim: int = -2) -> Tensor:
        attn_score = softmax(attn, index, ptr, dim_size, dim)
        attn = attn_score*x
        return self.reduce(attn, index, ptr, dim_size, dim, reduce='sum')