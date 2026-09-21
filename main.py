import logging

import hydra
import torch
from omegaconf import DictConfig

from Model.MoE import MoE
from Training import Trainer
from Utils.dati import carica_grafi
from Utils.logging_config import setup_logging
import Utils.run_dir  # registra il resolver indice_run usato in conf/config.yaml
from Utils.tracking import avvia_mlflow
import Training_checkpoint


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_logging(cfg.paths.log_file)
    logger = logging.getLogger("main")
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x, train, val, test, _ = carica_grafi(cfg.paths.dati, tuple(cfg.training.split))
    
    x = x.to(device)
    logger.info("grafi: %d train, %d val, %d test | %d nodi, %d feature | device %s",
                len(train), len(val), len(test), x.size(0), x.size(1), device)

    modello = MoE(in_channels=x.size(1),
                  num_communities=cfg.model.num_communities,
                  dim=cfg.model.dim, hidden=cfg.model.hidden,
                  dropout=cfg.model.dropout, hops=cfg.model.hops,
                  conv=cfg.model.conv, aggr=cfg.model.aggr)
    optim = torch.optim.Adam(modello.parameters(), lr=cfg.training.lr,
                             weight_decay=cfg.training.weight_decay)

    Training_checkpoint.set_checkpoint_dir(cfg.paths.checkpoint_dir)

    with avvia_mlflow(cfg) as run:
        trainer = Trainer(modello, optim, x, num_epochs=cfg.training.num_epochs,
                          grafi_per_step=cfg.training.grafi_per_step,
                          beta=cfg.training.beta, gamma=cfg.training.gamma,
                          tau=cfg.training.tau,
                          checkpoint_monitor=cfg.training.checkpoint_monitor,
                          checkpoint_mode=cfg.training.checkpoint_mode,
                          patience=cfg.training.patience,
                          grad_clip=cfg.training.grad_clip,
                          device=device, log_every=cfg.training.log_every,
                          mlflow_attivo=run is not None,
                          lr_factor=cfg.training.lr_factor,
                          lr_patience=cfg.training.lr_patience)
        best = trainer.train(train, val)
        trainer.salva_storico(cfg.paths.storico)

        if run is not None:
            import mlflow
            mlflow.log_metric("best_" + cfg.training.checkpoint_monitor, best)

    logger.info("fine training. %s migliore = %.4f all'epoca %s. Storico in %s",
                cfg.training.checkpoint_monitor, best, trainer.best_epoch,
                cfg.paths.storico)


if __name__ == "__main__":
    main()
