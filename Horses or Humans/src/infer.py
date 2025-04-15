import logging
import os

import hydra
import omegaconf
import torch
from inference.inference_pipeline import InferencePipeline
from utils.general_utils import setup_logging
from utils.seed_utils import fix_seed


@hydra.main(version_base=None, config_path="../conf", config_name="inference.yaml")
def main(cfg: omegaconf.DictConfig):
    logger = logging.getLogger(__name__)
    logger.info("Setting up logging configuration.\n")
    setup_logging(
        logging_config_path=os.path.join(
            hydra.utils.get_original_cwd(), "conf", "logging.yaml"
        )
    )

    if cfg.environ.seed:
        logger.info(f"Random seed set to {cfg.environ.seed}.\n")
        fix_seed(seed=cfg.environ.seed)

    if cfg.environ.device < 0:
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{cfg.environ.device}")
    logger.info(f"Device set to {device} for inference.\n")

    inference_pipeline = InferencePipeline(cfg=cfg, logger=logger, device=device)
    inference_pipeline.run_inference()


if __name__ == "__main__":
    main()
