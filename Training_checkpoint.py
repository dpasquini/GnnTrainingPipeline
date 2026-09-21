import os
from typing import Dict, Optional

import torch

CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")


def set_checkpoint_dir(path: str) -> str:
    global CHECKPOINT_DIR
    CHECKPOINT_DIR = os.path.abspath(os.path.expanduser(path))
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    return CHECKPOINT_DIR


def _percorso(filename: str) -> str:
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    return os.path.join(CHECKPOINT_DIR, filename)


def salva(model, optimizer, epoch: int, metrics: Dict[str, float],
          monitor: str, mode: str = "max",
          filename: str = "best_model.pt") -> None:
    """Scrive il checkpoint. Decidere SE salvare spetta al chiamante."""
    payload = {"epoch": epoch, "model_state_dict": model.state_dict(),
               "optimizer_state_dict": optimizer.state_dict(),
               "metrics": dict(metrics), "monitor": monitor, "mode": mode}

    finale = _percorso(filename)
    tmp = finale + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, finale)   # scrittura atomica: niente checkpoint troncati


def load_checkpoint(model, optimizer=None, device: str = "cpu",
                    filename: str = "best_model.pt", strict: bool = True) -> dict:
    """weights_only=False perche' il payload contiene anche il dict delle
    metriche: da torch 2.6 il default e' True e il load fallirebbe."""
    path = _percorso(filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint non trovato: {path}")

    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"], strict=strict)
    if optimizer is not None and state.get("optimizer_state_dict"):
        optimizer.load_state_dict(state["optimizer_state_dict"])
    return state
