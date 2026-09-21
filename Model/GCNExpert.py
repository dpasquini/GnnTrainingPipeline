import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv

CONVS = {"graphconv": GraphConv}


class GCNExpert(nn.Module):

    def __init__(self, in_channels: int, hidden: int = None,
                 out_channels: int = None, dropout: float = 0.3,
                 hops: int = 3, conv: str = "graphconv", aggr: str = "add"):
        super().__init__()
        hidden = hidden or in_channels
        out_channels = out_channels or hidden

        self.dropout = dropout
        dims = [in_channels] + [hidden] * (hops - 1) + [out_channels]
        Conv = CONVS[conv]
        self.convs = nn.ModuleList(
            [Conv(dims[i], dims[i + 1], aggr=aggr) for i in range(hops)])
        self.norms = nn.ModuleList([nn.LayerNorm(d) for d in dims[1:]])

    def forward(self, x, edge_index, edge_weight=None):
        h, ultimo = x, len(self.convs) - 1
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h = norm(conv(h, edge_index, edge_weight))
            if i < ultimo:
                h = F.dropout(F.relu(h), p=self.dropout, training=self.training)
        return h
