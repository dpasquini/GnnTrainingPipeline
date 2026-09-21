import contextlib
import glob
import hashlib
import logging
import os

from omegaconf import OmegaConf

logger = logging.getLogger("tracking")


def _hash_codice():
    
    radice = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sorgenti = sorted(glob.glob(os.path.join(radice, "Model", "*.py")))
    sorgenti += [os.path.join(radice, f) for f in ("Loss.py", "Training.py")]
    h = hashlib.sha1()
    for f in sorgenti:
        h.update(open(f, "rb").read())
    return h.hexdigest()[:12]


@contextlib.contextmanager
def avvia_mlflow(cfg):
    
    if not cfg.mlflow.enabled:
        yield None
        return

    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")
    os.environ.setdefault("DO_NOT_TRACK", "true")

    import mlflow
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment)

    with mlflow.start_run(run_name=cfg.mlflow.run_name) as run:
        params = {f"{sez}.{k}": v
                  for sez in ("model", "training")
                  for k, v in OmegaConf.to_container(cfg[sez]).items()}
        params["seed"] = cfg.seed
        params["code_hash"] = _hash_codice()
        mlflow.log_params(params)
        logger.info("MLflow run %s in %s", run.info.run_id, cfg.mlflow.tracking_uri)
        yield run
