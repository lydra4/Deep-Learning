import logging
import os

import hydra

from embeddings.perform_embeddings import PerformEmbeddings
from utils.general_utils import setup_logging
from utils.process_data import EPUBProcessor


@hydra.main(version_base=None, config_path="../conf", config_name="training.yaml")
def main(cfg):
    """
    Main function for executing the embedding pipeline.

    This function:
    - Sets up logging configuration.
    - Processes EPUB documents using `EPUBProcessor`.
    - Generates embeddings and stores them in a vector database using `PerformEmbeddings`.

    Args:
        cfg: A Hydra configuration object containing all necessary parameters.
    """
    logger = logging.getLogger(__name__)
    logger.info("Setting up logging configuration.\n")
    setup_logging(
        logging_config_path=os.path.join(
            hydra.utils.get_original_cwd(), "conf", "logging.yaml"
        )
    )

    epub_processor = EPUBProcessor(cfg=cfg, logger=logger)
    documents = epub_processor.load()

    perform_embeddings = PerformEmbeddings(cfg=cfg, logger=logger, documents=documents)
    perform_embeddings.generate_vectordb()


if __name__ == "__main__":
    main()
