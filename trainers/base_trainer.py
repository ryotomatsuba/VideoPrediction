# -*- coding: utf-8 -*-
"""Abstract base model"""
import glob
import logging
from abc import ABC
from omegaconf import DictConfig,ListConfig
from mlflow.tracking import MlflowClient
import mlflow
import torch
from torch.nn.modules import loss

log = logging.getLogger(__name__)
class MlflowWriter():
    """wrapper class for logging to mlflow from anywere"""
    def __init__(self, experiment_name, mlrun_path, **kwargs):
        mlflow.set_tracking_uri(mlrun_path)
        self.client = MlflowClient(**kwargs)
        try:
            self.experiment_id = self.client.create_experiment(experiment_name)
        except:
            self.experiment_id = self.client.get_experiment_by_name(experiment_name).experiment_id

        self.run_id = self.client.create_run(self.experiment_id).info.run_id

    def log_params_from_omegaconf_dict(self, params):
        for param_name, element in params.items():
            self._explore_recursive(param_name, element)

    def _explore_recursive(self, parent_name, element):
        if isinstance(element, DictConfig):
            for k, v in element.items():
                if isinstance(v, DictConfig) or isinstance(v, ListConfig):
                    self._explore_recursive(f'{parent_name}.{k}', v)
                else:
                    self.client.log_param(self.run_id, f'{parent_name}.{k}', v)
        elif isinstance(element, ListConfig):
            for i, v in enumerate(element):
                self.client.log_param(self.run_id, f'{parent_name}.{i}', v)

    def log_torch_model(self, model):
        with mlflow.start_run(self.run_id):
            torch.log_model(model, 'models')

    def log_params(self, params:dict):
        for key,value in params.items():
            self.client.log_param(self.run_id, key, value)
    def log_metrics(self, metrics:dict, step: int = None):
        for key,value in metrics.items():
            self.client.log_metric(self.run_id, key, value, step = step)

    def log_artifact(self, local_path):
        self.client.log_artifact(self.run_id, local_path)

    def set_terminated(self):
        self.client.set_terminated(self.run_id)


class BaseTrainer(ABC):
    """Abstract base trainer

    This model is inherited by all trainers.

    Attributes:
        cfg: Config of project.

    """

    def __init__(self, cfg: object) -> None:
        """Initialization

        Args:
            cfg: Config of project.

        """
        self.mlwriter=MlflowWriter(cfg.experiment.name,cfg.mlrun_path)
        self.loss={"train": [], "val": []}
        self.cfg = cfg
        self.log_params()


    def execute(self, eval: bool) -> None:
        """Execution

        Execute train or eval.

        Args:
            eval: For evaluation mode.
                True: Execute eval.
                False: Execute train.

        """
        pass


    def train(self) -> None:
        """Train

        Trains model.

        """

        log.info("Training process has begun.")
        

    def eval(self,eval_dataloader: object = None, epoch: int = 0) -> float:
        """Evaluation

        Evaluates model.

        Args:
            eval_dataloader: Dataloader.
            epoch: Number of epoch.

        Returns:
            model_score: Indicator of the excellence of model. The higher the value, the better.

        """
        
        log.info('Evaluation:')


    def log_params(self) -> None:
        """Log parameters"""
        
        self.mlwriter.log_params_from_omegaconf_dict(self.cfg)
        self.mlwriter.log_artifact(".hydra/config.yaml")


    def log_base_artifacts(self) -> None:
        """log artifacts"""
        self.mlwriter.log_artifact(glob.glob(r"*.log")[0])
        self.mlwriter.log_artifact(self.cfg.train.ckpt_path)
    
    def log_artifact(self,path) -> None:
        """log artifacts"""
        self.mlwriter.log_artifact(path)
    
    def log_metrics(self,epoch) -> None:
        metrics={
            "train_loss":self.loss["train"][-1],
            "val_loss":self.loss["val"][-1],
        }
        self.mlwriter.log_metrics(metrics, step=epoch)