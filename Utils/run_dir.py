import os

from omegaconf import OmegaConf


def _indice(dir_giorno):
    """Progressivo del run dentro la cartella del giorno: max esistente + 1."""
    esistenti = ([int(n) for n in os.listdir(dir_giorno) if n.isdigit()]
                 if os.path.isdir(dir_giorno) else [])
    return str(max(esistenti, default=-1) + 1)



OmegaConf.register_new_resolver("indice_run", _indice, use_cache=True)
