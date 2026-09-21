from typing import Optional, Tuple

import torch
from torch_geometric.utils import coalesce, remove_self_loops


def build_modularity_terms(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_weight: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """A, k_out, k_in, m del grafo diretto: self loop rimossi, archi duplicati
    sommati. A resta sparsa, B non viene mai materializzata."""
    device = device or edge_index.device
    edge_index = edge_index.to(device=device, dtype=torch.long)

    if edge_weight is not None:
        edge_weight = edge_weight.to(device=device, dtype=dtype)

    edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), device=device, dtype=dtype)

    edge_index, edge_weight = coalesce(edge_index, edge_weight,
                                       num_nodes=num_nodes, reduce="sum")

    A = torch.sparse_coo_tensor(edge_index, edge_weight,
                                size=(num_nodes, num_nodes), device=device).coalesce()

    k_out = torch.sparse.sum(A, dim=1).to_dense()
    k_in = torch.sparse.sum(A, dim=0).to_dense()
    m = edge_weight.sum()

    return A, k_out, k_in, m


def modularity_loss(S, A, k_out, k_in, m, eps: float = 1e-9):
    """
    Modularita' diretta (Leicht-Newman), rilassamento continuo, senza mai
    materializzare B:

        Tr(S^T B S) = Tr(S^T A S) - (1/m) * (S^T k_out) . (S^T k_in)
    """
    if S.dtype != A.dtype:
        S = S.to(A.dtype)

    tr_SAS = (S * torch.sparse.mm(A, S)).sum()
    Sk_out = S.t() @ k_out
    Sk_in = S.t() @ k_in
    tr_atteso = (Sk_out * Sk_in).sum() / (m + eps)

    return -(tr_SAS - tr_atteso) / (m + eps)


@torch.no_grad()
def hard_modularity(S, A, k_out, k_in, m, eps: float = 1e-9) -> float:
    """
    Modularita' sulla partizione discreta (argmax di S). E' la metrica che
    descrive davvero il risultato: Q sul rilassamento continuo puo' salire
    mentre la partizione hard peggiora. Non differenziabile.
    """
    hard = torch.zeros_like(S)
    hard.scatter_(1, S.argmax(dim=1, keepdim=True), 1.0)

    tr_SAS = (hard * torch.sparse.mm(A, hard)).sum()
    tr_atteso = ((hard.t() @ k_out) * (hard.t() @ k_in)).sum() / (m + eps)
    return float((tr_SAS - tr_atteso) / (m + eps))


@torch.no_grad()
def community_stats(S) -> dict:
    """Il collasso su 1-2 community e' il fallimento tipico di questo training
    e va visto subito nei log."""
    counts = torch.bincount(S.argmax(dim=1), minlength=S.size(1)).float()
    frac = counts / counts.sum().clamp(min=1)
    nonzero = frac[frac > 0]
    return {
        "communities_used": int((counts > 0).sum()),
        "largest_community_frac": float(frac.max()),
        "assignment_entropy": float(-(nonzero * nonzero.log()).sum()),
    }


def community_balance_loss(S) -> torch.Tensor:
    """
    Anti-collasso in stile DMoN: sqrt(K)/N * ||sum_i S_i||_2 - 1.
    Vale 0 con massa uniforme sulle K community, cresce fino a sqrt(K)-1
    quando finiscono tutti nella stessa.
    """
    num_nodes, num_communities = S.shape
    return (num_communities ** 0.5) / num_nodes * torch.norm(S.sum(dim=0), p=2) - 1.0


def total_loss(moe, x, edge_index_dict, edge_weight_dict, A, k_out, k_in, m,
               beta: float = 1.0, gamma: float = 1.0, tau: float = 1.0,
               compute_hard: bool = True) -> Tuple[torch.Tensor, dict]:
    """
    Un forward piu' la loss completa. A, k_out, k_in e m vanno costruiti
    sull'unione dei canali degli STESSI nodi di x, cioe' sul grafo orario.
    """
    S, alpha, riceve = moe.community_assignment(x, edge_index_dict,
                                                edge_weight_dict, tau)

    l_mod = modularity_loss(S, A, k_out, k_in, m)
    l_exp = moe.load_balance_loss(alpha, riceve)
    l_bal = community_balance_loss(S)
    loss = l_mod + beta * l_exp + gamma * l_bal

    metrics = {
        "Q_routing": float(-l_mod.detach()),
        "L_expert": float(l_exp.detach()),
        "L_community_balance": float(l_bal.detach()),
        "loss_total": float(loss.detach()),
    }
    if compute_hard:
        metrics["Q_hard"] = hard_modularity(S.detach(), A, k_out, k_in, m)
        metrics.update(community_stats(S.detach()))
       
        liberi = riceve.all(dim=1)
        metrics["nodi_gate"] = int(liberi.sum())
        if metrics["nodi_gate"]:
            a = alpha.detach()[liberi]
            for ch, val in zip(moe.channels, a.mean(dim=0).tolist()):
                metrics[f"alpha_{ch}"] = val
        
            metrics["alpha_entropy"] = float(-(a * (a + 1e-9).log()).sum(dim=1).mean())

    return loss, metrics
