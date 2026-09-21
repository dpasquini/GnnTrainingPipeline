import torch
import torch.nn as nn

from .GCNExpert import GCNExpert


CHANNELS = ('retweet', 'response')


class Router(nn.Module):

    def __init__(self, in_channels: int, dim: int, num_communities: int,
                 hidden: int = None, dropout: float = 0.3, hops: int = 3,
                 conv: str = "graphconv", aggr: str = "add"):
        super().__init__()
        if num_communities < 2:
            raise ValueError("num_communities deve essere >= 2: con una sola "
                             "community la modularita' e' identicamente 0")

        self.channels = CHANNELS

        self.input_proj = nn.Linear(in_channels, dim)
        self.experts = nn.ModuleDict({
            ch: GCNExpert(dim, hidden or dim, dim, dropout, hops, conv, aggr)
            for ch in CHANNELS
        })
        self.gate = nn.Linear(dim * len(CHANNELS), len(CHANNELS))
        self.community_head = nn.Linear(dim, num_communities)

    def _maschera(self, edge_index_dict, num_nodes, device):
        
        masks = []
        for ch in self.channels:
            ei = edge_index_dict.get(ch)
            active = torch.zeros(num_nodes, dtype=torch.bool, device=device)
            if ei.numel() > 0:
                active[ei[1]] = True
            masks.append(active)

        return torch.stack(masks, dim=1)

    def forward(self, x, edge_index_dict, edge_weight_dict=None):
       
        num_nodes, device = x.size(0), x.device
        edge_weight_dict = edge_weight_dict or {}

        vuoto = torch.empty((2, 0), dtype=torch.long, device=device)
        ei = {ch: edge_index_dict.get(ch, vuoto) for ch in self.channels}

        h = self.input_proj(x)
        z_list = [self.experts[ch](h, ei[ch], edge_weight_dict.get(ch))
                  for ch in self.channels]

        riceve = self._maschera(ei, num_nodes, device)
        aggrega = riceve.any(dim=1, keepdim=True)
        logits = self.gate(torch.cat(z_list, dim=1))

        logits = logits.masked_fill(~(riceve | ~aggrega), float('-inf'))
        alpha = torch.softmax(logits, dim=1)
        alpha = torch.where(aggrega, alpha,
                            torch.full_like(alpha, 1.0 / len(self.channels)))

        z = h + (alpha.unsqueeze(-1) * torch.stack(z_list, dim=1)).sum(dim=1)
        return z, alpha, riceve

    def community_assignment(self, x, edge_index_dict, edge_weight_dict=None,
                             tau: float = 1.0):
        z, alpha, riceve = self.forward(x, edge_index_dict, edge_weight_dict)
        S = torch.softmax(self.community_head(z) / tau, dim=1)
        return S, alpha, riceve
