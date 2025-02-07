import logging
import os

import hydra
from embeddings.perform_embeddings import PerformEmbeddings
from utils.general_utils import setup_logging
from utils.process_data import EPUBProcessor


@hydra.main(version_base=None, config_path="../conf", config_name="training.yaml")
def main(cfg):
    logger = logging.getLogger(__name__)
    logger.info("Setting up logging configuration.")
    setup_logging(
        logging_config_path=os.path.join(
            hydra.utils.get_original_cwd(), "conf", "logging.yaml"
        )
    )

    epub_processor = EPUBProcessor(cfg=cfg, logger=logger)
    documents = epub_processor.load()

    perform_embeddings = PerformEmbeddings(cfg=cfg, logger=logger, documents=documents)
    vector_db = perform_embeddings.process_and_load_vectordb()


if __name__ == "__main__":
    main()
