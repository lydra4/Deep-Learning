import logging
import os

import hydra

from inference.inference_pipeline import InferencePipeline
from utils.general_utils import setup_logging


@hydra.main(version_base=None, config_path="../conf", config_name="inference.yaml")
def main(cfg):
    """
    Main function for executing the inference pipeline.

    This function:
    - Sets up logging configuration.
    - Initializes the `InferencePipeline` with the given configuration.
    - Runs the inference process.

    Args:
        cfg: A Hydra configuration object containing all necessary parameters.
    """
    logger = logging.getLogger(__name__)
    logger.info("Setting up logging configuration")
    setup_logging(
        logging_config_path=os.path.join(
            hydra.utils.get_original_cwd(), "conf", "logging.yaml"
        )
    )

    infer_pipeline = InferencePipeline(cfg=cfg, logger=logger)
    infer_pipeline.run_inference()


if __name__ == "__main__":
    main()
