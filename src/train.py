import logging
import os

import hydra
from langchain_community.document_loaders import DirectoryLoader

from utils.general_utils import setup_logging


@hydra.main(version_base=None, config_path="../conf", config_name="training.yaml")
def main(cfg):
    logger = logging.getLogger(__name__)
    logger.info("Setting up logging configuration.")
    setup_logging(
        logging_config_path=os.path.join(
            hydra.utils.get_original_cwd(), "conf", "logging.yaml"
        )
    )

    epub_loader = DirectoryLoader(
        path=cfg["preprocessing"]["path"],
        glob=cfg["preprocessing"]["glob"],
        show_progress=cfg["preprocessing"]["show_progress"],
        use_multithreading=cfg["preprocessing"]["use_multithreading"],
        sample_seed=cfg["preprocessing"]["sample_seed"],
    )

    logger.info("Loading Documents")
    docs = epub_loader.load()


if __name__ == "__main__":
    main()
