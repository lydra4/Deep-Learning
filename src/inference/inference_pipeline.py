import logging

import omegaconf
import torch
from training_pipelines.training_pipelines import TrainingPipeline


class InferencePipeline(TrainingPipeline):
    def __init__(
        self, cfg: omegaconf.DictConfig, logger: logging.Logger, device: torch.device
    ) -> None:
        """
        Initializes the InferencePipeline.

        Args:
            cfg (omegaconf.DictConfig): Configuration object with dataset and environment settings.
            logger (logging.Logger): Logger instance for logging.
            device (torch.device): Device to use for inference (CPU or GPU).
        """
        super().__init__(cfg=cfg, logger=logger, device=device)
