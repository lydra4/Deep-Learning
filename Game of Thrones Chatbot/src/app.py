import logging
import os

import hydra
from frontend.gradio_app import GradioApp
from utils.general_utils import setup_logging


@hydra.main(version_base=None, config_path="../conf", config_name="inference.yaml")
def main(cfg):
    logger = logging.getLogger(__name__)
    logger.info("Setting up logging configuration")
    setup_logging(
        logging_config_path=os.path.join(
            hydra.utils.get_original_cwd(), "conf", "logging.yaml"
        )
    )

    gradio_app = GradioApp(cfg=cfg, logger=logger)
    gradio_app.launch_app()


if __name__ == "__main__":
    main()
