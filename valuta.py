
import csv
import logging

import hydra
import torch
from omegaconf import DictConfig

import Loss
import Training_checkpoint
from Model.MoE import MoE
from Training import prepara
from Utils.dati import carica_grafi
from Utils.logging_config import setup_logging
import Utils.run_dir


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_logging(cfg.paths.log_file)
    logger = logging.getLogger("valuta")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x, _, _, test, _ = carica_grafi(cfg.paths.dati, tuple(cfg.training.split))
    x = x.to(device)

    modello = MoE(in_channels=x.size(1),
                  num_communities=cfg.model.num_communities,
                  dim=cfg.model.dim, hidden=cfg.model.hidden,
                  dropout=cfg.model.dropout, hops=cfg.model.hops,
                  conv=cfg.model.conv, aggr=cfg.model.aggr).to(device)

    Training_checkpoint.set_checkpoint_dir(cfg.paths.checkpoint_dir)
    stato = Training_checkpoint.load_checkpoint(modello, device=device)
    logger.info("checkpoint dell'epoca %s, %s", stato["epoch"], stato["metrics"])

    modello.eval()
    righe = []
    with torch.no_grad():
        for g in test:
            x_t, ei, ew, A, k_out, k_in, m = prepara(g, x, device)
            if float(m) <= 0:
                continue
            S, _, _ = modello.community_assignment(x_t, ei, ew, tau=cfg.training.tau)
            righe.append({"ts": g["ts"], "nodi": x_t.size(0),
                          "archi": sum(e.size(1) for e in ei.values()),
                          "Q_hard": Loss.hard_modularity(S, A, k_out, k_in, m),
                          **Loss.community_stats(S)})

    q = torch.tensor([r["Q_hard"] for r in righe])
    logger.info("%d grafi di test | Q_hard media %.4f (dev %.4f, min %.4f, max %.4f)",
                len(righe), q.mean(), q.std(), q.min(), q.max())
    logger.info("communities_used media %.1f | piu' grande media %.4f",
                sum(r["communities_used"] for r in righe) / len(righe),
                sum(r["largest_community_frac"] for r in righe) / len(righe))

    with open(cfg.paths.assegnazioni, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(righe[0]))
        w.writeheader()
        w.writerows(righe)
    logger.info("dettaglio per grafo in %s", cfg.paths.assegnazioni)


if __name__ == "__main__":
    main()
