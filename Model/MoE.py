import torch
import torch.nn as nn

from .Router import Router, CHANNELS


class MoE(nn.Module):

    def __init__(self, in_channels: int, num_communities: int,
                 dim: int = 64, hidden: int = None, dropout: float = 0.3,
                 hops: int = 3, conv: str = "graphconv", aggr: str = "add"):
        super().__init__()
        self.channels = CHANNELS

        self.router = Router(in_channels, dim, num_communities, hidden,
                             dropout, hops, conv, aggr)

    def forward(self, x, edge_index_dict, edge_weight_dict=None):
        return self.router(x, edge_index_dict, edge_weight_dict)

    def community_assignment(self, x, edge_index_dict, edge_weight_dict=None,
                             tau: float = 1.0):
        return self.router.community_assignment(x, edge_index_dict,
                                                edge_weight_dict, tau)

    @staticmethod
    def load_balance_loss(alpha: torch.Tensor, riceve: torch.Tensor,
                          eps: float = 1e-9) -> torch.Tensor:
        
        validi = riceve.all(dim=1)
        if not bool(validi.any()):
            return torch.zeros((), device=alpha.device, dtype=alpha.dtype)

        alpha = alpha[validi]
        disp = riceve[validi].float()

        target = disp / disp.sum(dim=1, keepdim=True).clamp_min(1.0)
        importance = alpha.mean(dim=0)
        target_importance = target.mean(dim=0)

        attivi = target_importance > 0
        errore = ((importance[attivi] - target_importance[attivi])
                  / (target_importance[attivi] + eps))
        return errore.pow(2).mean()
