from typing import Optional
import torch 
from torch import Tensor
from torch_geometric.nn.pool.connect import FilterEdges
from torch_geometric.nn.pool.select.topk import topk
from torch_geometric.nn.pool.select import SelectOutput
from torch_geometric.utils import add_remaining_self_loops
from .MP.LSCMP import LSCMP
from .MP.LSSMP import LSSMP

class LSPooling(torch.nn.Module):
    def __init__(
            self,
            in_channels: int,
            ratio: float = 0.5,
            nonlinearity: str = "elu",
            act: str = "tanh",
            *args, 
            **kwargs
        ):
        super().__init__(*args, **kwargs)
        self.in_channels = in_channels
        # produce the importance score for each local set
        self.lssmp = LSSMP(in_channels, nonlinearity=nonlinearity, act=act, *args, **kwargs)
        # filter local sets with topk scores
        self.connect = FilterEdges()
        # reduce local set with local set collapse MP
        self.lscmp = LSCMP()
        self.ratio = ratio
        self.reset_parameters()

    def reset_parameters(self):
        self.lssmp.reset_parameters()
        self.lscmp.reset_parameters()

    def forward(
            self,
            x: Tensor,
            edge_index: Tensor,
            edge_attr: Optional[Tensor]=None,
            batch: Tensor=None,
    ):
        # The situation batch size is 1?
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        
        # add self-loop
        edge_index, edge_attr = add_remaining_self_loops(edge_index, edge_attr)

        # select returns dense select out
        scores = self.lssmp.forward(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )
        scores = scores.squeeze(-1)
        assert scores is not None
        
        node_index = topk(scores, ratio=self.ratio, batch=batch)

        select_output= SelectOutput(
            node_index = node_index,
            num_nodes = x.size(0),
            cluster_index = torch.arange(node_index.size(0), device=x.device),
            num_clusters = node_index.size(0),
            weight=scores[node_index]
        )

        

        # local set collapse
        x = self.lscmp(x, scores.unsqueeze(-1), edge_index)

        # filter nodes and edges
        x = x[node_index]
        # use A^2 to keep the connectivity
        connect_out = self.connect.forward(select_output, edge_index, edge_attr, batch)

        return (x, connect_out.edge_index, connect_out.edge_attr, connect_out.batch, node_index, scores)
    