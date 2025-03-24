import logging
import os

import hydra
from data_prep.pipelines import ImagePipeline
from utils.general_utils import setup_logging
from utils.seed_utils import fix_seed


@hydra.main(version_base=None, config_path="../conf", config_name="data_processing")
def main(cfg):
    """Main function to set up logging, fix random seed, and execute the image processing pipeline.

    This function performs the following tasks:
    1. Initializes logging based on the specified configuration.
    2. Fixes the random seed for reproducibility.
    3. Instantiates the `ImagePipeline` class and executes the `run_pipeline()` method.

    Args:
        cfg (dict): Configuration dictionary containing paths, seed, and augmentation settings.

    """
    logger = logging.getLogger(__name__)
    logger.info("Setting up logging configuration.\n")
    setup_logging(
        logging_config_path=os.path.join(
            hydra.utils.get_original_cwd(), "conf", "logging.yaml"
        )
    )
    fix_seed(seed=cfg["seed"])
    logger.info(f"Seed fixed at {cfg['seed']}\n")
    image_pipeline = ImagePipeline(cfg=cfg, logger=logger)
    image_pipeline.run_pipeline()


if __name__ == "__main__":
    main()
