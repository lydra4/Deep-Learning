import logging
import os

import hydra
from training_pipelines.training_pipelines import TrainingPipeline
from utils.general_utils import setup_logging
from utils.seed_utils import fix_seed


@hydra.main(version_base=None, config_path="../conf", config_name="training.yaml")
def main(cfg):
    """
    Main function to set up the logging configuration, fix the seed for reproducibility,
    and initialize and run the training pipeline.

    This function is decorated with Hydra's @hydra.main to provide configuration management.
    It sets up logging according to the provided logging configuration file, fixes the
    random seed for reproducibility, and then creates and runs a training pipeline
    using the specified configuration.

    Args:
        cfg (dict): The configuration dictionary provided by Hydra, containing various
                    configuration options for training, including the seed value.

    Returns:
        None
    """
    logger = logging.getLogger(__name__)
    logger.info("Setting up logging configuration.")
    setup_logging(
        logging_config_path=os.path.join(
            hydra.utils.get_original_cwd(), "conf", "logging.yaml"
        )
    )

    fix_seed(seed=cfg["seed"])
    logger.info(f"Seed fixed at {cfg['seed']}.\n")

    training_pipeline = TrainingPipeline(cfg=cfg, logger=logger)
    training_pipeline.run_training_pipeline()

    os.system("shutdown /s /t 60")


if __name__ == "__main__":
    main()
