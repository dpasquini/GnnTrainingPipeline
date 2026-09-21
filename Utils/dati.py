import torch


def carica_grafi(path, split=(0.7, 0.15, 0.15)):
    """
    Carica dati_orari.pt e divide i grafi in ordine TEMPORALE.

    """
    D = torch.load(path, weights_only=False)
    grafi = D["grafi"]
    n = len(grafi)
    a = int(n * split[0])
    b = a + int(n * split[1])
    return D["x"], grafi[:a], grafi[a:b], grafi[b:], D
