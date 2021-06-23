import hydra
from omegaconf import DictConfig
import mlflow
import os
import logging
logger = logging.getLogger(__name__)

def train(cfg):
    for epoch in range(1,cfg.train.epochs+1):
        logger.info(f'epoch={epoch}')
        metrics={
            "loss_train":epoch+1,
            "loss_val":epoch+2,
        }
        mlflow.log_metrics(metrics,step=epoch)


@hydra.main(config_path="configs", config_name="default")
def main(cfg: DictConfig) -> None:
    cwd=hydra.utils.get_original_cwd()
    mlrun_path=f'/data/{cfg.output_dir}/mlruns/'
    if not os.path.isdir(mlrun_path):
        os.makedirs(mlrun_path, exist_ok=True)
    mlflow.set_tracking_uri(mlrun_path)
    mlflow.set_experiment(cfg.experiment.name)
    with mlflow.start_run():
        mlflow.log_params({"name":cfg.experiment.name})
        mlflow.log_artifact(".hydra/config.yaml")
        train(cfg)
        mlflow.log_artifact("train.log")
if __name__=="__main__":
    main()