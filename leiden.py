
import csv
import logging

import hydra
import igraph as ig
import torch
from omegaconf import DictConfig

import Loss
from Model.Router import CHANNELS
from Utils.dati import carica_grafi
from Utils.logging_config import setup_logging
import Utils.run_dir


def q_diretta(src, dst, w, etichette, k_out, k_in, m):
    intra = w[etichette[src] == etichette[dst]].sum()
    n_com = int(etichette.max()) + 1
    K_out = torch.zeros(n_com, dtype=torch.float64).index_add_(0, etichette, k_out.double())
    K_in = torch.zeros(n_com, dtype=torch.float64).index_add_(0, etichette, k_in.double())
    return float(intra.double() / m - (K_out * K_in).sum() / m ** 2)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_logging(cfg.paths.log_file)
    logger = logging.getLogger("leiden")

    K = cfg.model.num_communities
    _, _, _, test, _ = carica_grafi(cfg.paths.dati, tuple(cfg.training.split))
    righe = []

    for g in test:
        ei = torch.cat([g[ch][0].long() for ch in CHANNELS], dim=1)
        w = torch.cat([g[ch][1] for ch in CHANNELS])
        n = g["nodi"].numel()
        A, k_out, k_in, m = Loss.build_modularity_terms(ei, n, w)
        if float(m) <= 0:
            continue
        src, dst, peso = A.indices()[0], A.indices()[1], A.values()

        gr = ig.Graph(n=n, edges=list(zip(src.tolist(), dst.tolist())), directed=False)
        pesi = peso.tolist()
        # n_iterations=-1: itera fino a convergenza, e' il regime in cui Leiden
        # garantisce comunita' connesse
        part = gr.community_leiden(objective_function="modularity",
                                   weights=pesi, n_iterations=-1)
        et = torch.as_tensor(part.membership, dtype=torch.long)
        conteggi = torch.bincount(et)
        taglie = conteggi.sort(descending=True).values

        # troncamento: le K-1 piu' grandi tengono il proprio indice, il resto
        # finisce tutto nella K-esima
        ordine = conteggi.argsort(descending=True)[:K - 1]
        mappa = torch.full((len(conteggi),), K - 1, dtype=torch.long)
        mappa[ordine] = torch.arange(len(ordine))
        et_k = mappa[et]

        righe.append({
            "ts": g["ts"], "nodi": n, "archi": peso.numel(),
            "comunita": len(part),
            "Q_diretta": q_diretta(src, dst, peso, et, k_out, k_in, m),
            "Q_non_diretta": gr.modularity(part.membership, weights=pesi),
            "frac_piu_grande": float(taglie[0]) / n,
            "Q_diretta_K": q_diretta(src, dst, peso, et_k, k_out, k_in, m),
            "frac_piu_grande_K": float(torch.bincount(et_k).max()) / n,
            "taglia_mediana": float(taglie.median()),
            "com_da_10_su": int((taglie >= 10).sum()),
            "frac_singoletti": float((taglie == 1).sum()) / len(part),
        })

    def media(campo):
        return sum(r[campo] for r in righe) / len(righe)

    q = torch.tensor([r["Q_diretta"] for r in righe])
    logger.info("%d grafi di test | Q diretta media %.4f (dev %.4f, min %.4f, max %.4f)",
                len(righe), q.mean(), q.std(), q.min(), q.max())
    logger.info("Q non diretta media %.4f (quella che Leiden massimizza)",
                media("Q_non_diretta"))
    logger.info("comunita' per grafo %.0f in media, di cui %.0f con almeno 10 nodi | "
                "singoletti %.1f%%",
                media("comunita"), media("com_da_10_su"), 100 * media("frac_singoletti"))
    logger.info("piu' grande %.1f%% dei nodi in media | taglia mediana %.1f nodi",
                100 * media("frac_piu_grande"), media("taglia_mediana"))

    qk = torch.tensor([r["Q_diretta_K"] for r in righe])
    logger.info("troncata a K=%d | Q diretta media %.4f (dev %.4f, min %.4f, max %.4f) "
                "| piu' grande %.1f%% (uniforme sarebbe %.1f%%)",
                K, qk.mean(), qk.std(), qk.min(), qk.max(),
                100 * media("frac_piu_grande_K"), 100 / K)

    with open(cfg.paths.leiden, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(righe[0]))
        w.writeheader()
        w.writerows(righe)
    logger.info("dettaglio per grafo in %s", cfg.paths.leiden)


if __name__ == "__main__":
    main()
