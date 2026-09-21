

import logging
import time
from typing import Dict, List, Optional

import torch

import Loss
import Training_checkpoint
from Model.Router import CHANNELS

logger = logging.getLogger("Trainer")

# ordine in cui compaiono nel log: prima quelle che si guardano davvero
PRINCIPALI = ("loss_total", "Q_hard", "Q_val", "Q_routing", "L_expert",
              "L_community_balance", "communities_used", "alpha_entropy",
              "nodi_gate")

# colonne dello storico su CSV, in quest'ordine
STORICO = ("epoch", "loss_total", "Q_hard", "Q_val", "Q_routing", "L_expert",
           "L_community_balance", "communities_used", "largest_community_frac",
           *(f"alpha_{ch}" for ch in CHANNELS), "alpha_entropy", "nodi_gate",
           "lr", "senza_miglioramenti")


def prepara(g, x, device):
    """Dal grafo salvato ai tensori sul device.

    I nodi del grafo sono indici globali nelle righe di x, gli edge_index sono
    gia' locali a quei nodi. Se x sta gia' sul device l'indicizzazione non
    trasferisce niente: e' il motivo per cui main.py ce la mette.
    """
    x_t = x[g["nodi"].long().to(x.device)].to(device)
    ei = {ch: g[ch][0].long().to(device) for ch in CHANNELS}
    ew = {ch: g[ch][1].to(device) for ch in CHANNELS}
    A, k_out, k_in, m = Loss.build_modularity_terms(
        torch.cat([ei[ch] for ch in CHANNELS], dim=1), x_t.size(0),
        torch.cat([ew[ch] for ch in CHANNELS]), device=device, dtype=x_t.dtype)
    return x_t, ei, ew, A, k_out, k_in, m


class Trainer:

    def __init__(self, model, optimizer, x, num_epochs: int,
                 grafi_per_step: int = 16,
                 beta: float = 1.0, gamma: float = 1.0, tau: float = 1.0,
                 checkpoint_monitor: str = "Q_val", checkpoint_mode: str = "max",
                 patience: Optional[int] = None, grad_clip: Optional[float] = 1.0,
                 device=None, log_every: int = 1,
                 mlflow_attivo: bool = False,
                 lr_factor: float = 0.5, lr_patience: Optional[int] = None):
        self.device = device or torch.device("cpu")
        self.model = model.to(self.device)
        self.x = x
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.grafi_per_step = max(1, int(grafi_per_step))
        self.beta, self.gamma, self.tau = beta, gamma, tau
        self.grad_clip = grad_clip
        self.log_every = max(1, int(log_every))
        self.mlflow_attivo = mlflow_attivo
        self.checkpoint_monitor = checkpoint_monitor
        self.checkpoint_mode = checkpoint_mode

        self.patience = patience
        self._best = float("-inf")
        self.best_epoch: Optional[int] = None
        self._t_epoca = self._t_valida = 0.0
        self.history: List[Dict[str, float]] = []

        self.sched = (torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=checkpoint_mode, factor=lr_factor,
            patience=lr_patience) if lr_patience else None)

    def _passata(self, grafi, allena: bool) -> Dict[str, float]:
        somma, n, nello_step = {}, 0, 0
        if allena:
            self.optimizer.zero_grad(set_to_none=True)

        for g in grafi:
            x_t, ei, ew, A, k_out, k_in, m = prepara(g, self.x, self.device)
            if float(m) <= 0:
                continue

            with torch.set_grad_enabled(allena):
                loss, metrics = Loss.total_loss(
                    self.model, x_t, ei, ew, A, k_out, k_in, m,
                    beta=self.beta, gamma=self.gamma, tau=self.tau)

            if allena:
                (loss / self.grafi_per_step).backward()
                nello_step += 1
                if nello_step == self.grafi_per_step:
                    if self.grad_clip:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                       self.grad_clip)
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    nello_step = 0

            for k, v in metrics.items():
                somma[k] = somma.get(k, 0.0) + v
            n += 1

        # ultimo step incompleto
        if allena and nello_step:
            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

        return {k: v / n for k, v in somma.items()} if n else {}

    def train(self, grafi_train, grafi_val) -> Optional[float]:
        senza_miglioramenti = 0

        for epoch in range(self.num_epochs):
            t = time.perf_counter()
            self.model.train()
            metrics = self._passata(grafi_train, allena=True)
            if not metrics:
                raise RuntimeError("Nessun grafo utilizzabile: tutti senza archi.")
            self._t_epoca = time.perf_counter() - t

            t = time.perf_counter()
            self.model.eval()
            val = self._passata(grafi_val, allena=False)
            self._t_valida = time.perf_counter() - t
            metrics["Q_val"] = val.get("Q_hard", float("nan"))
            metrics["lr"] = self.optimizer.param_groups[0]["lr"]

            if self.checkpoint_monitor not in metrics:
                raise KeyError(f"checkpoint_monitor={self.checkpoint_monitor!r} "
                               f"non e' fra {sorted(metrics)}")

            q = metrics[self.checkpoint_monitor]
            migliorato = q > self._best
            if migliorato:
                self._best = q
                self.best_epoch = epoch
                senza_miglioramenti = 0
                Training_checkpoint.salva(
                    self.model, self.optimizer, epoch, metrics,
                    monitor=self.checkpoint_monitor, mode=self.checkpoint_mode)
            else:
                senza_miglioramenti += 1

            self.history.append({"epoch": epoch, **metrics,
                                 "senza_miglioramenti": senza_miglioramenti})

            if self.mlflow_attivo:
                import mlflow
                mlflow.log_metrics({k: float(v) for k, v in metrics.items()},
                                   step=epoch)
            if self.sched:
                self.sched.step(metrics[self.checkpoint_monitor])

            if epoch % self.log_every == 0:
                logger.info("[epoca %d] %s | train %.0fs val %.0fs%s",
                            epoch, self._formatta(metrics), self._t_epoca,
                            self._t_valida,
                            "  <-- best" if migliorato else "")

            if self.patience and senza_miglioramenti >= self.patience:
                logger.info("[early stopping] nessun miglioramento su %r da %d "
                            "epoche (migliore: epoca %d). Stop all'epoca %d.",
                            self.checkpoint_monitor, self.patience,
                            self.best_epoch, epoch)
                break

        return self._best

    @staticmethod
    def _formatta(metrics: dict) -> str:
        ordinate = ([k for k in PRINCIPALI if k in metrics]
                    + [k for k in metrics if k not in PRINCIPALI])
        return " ".join(f"{k}={metrics[k]:.4f}" for k in ordinate)

    def salva_storico(self, path: str) -> None:
        """Una riga per epoca, solo le colonne di STORICO."""
        import csv, os
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        campi = [k for k in STORICO if k in self.history[0]]
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=campi, extrasaction="ignore")
            w.writeheader()
            w.writerows(self.history)
