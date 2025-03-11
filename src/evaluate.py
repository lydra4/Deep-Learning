import logging
import os

import hydra
import omegaconf

from evaluation.evaluation_pipeline import EvaluationPipeline
from utils.general_utils import setup_logging


@hydra.main(version_base=None, config_path="../conf", config_name="evaluate.yaml")
def main(cfg: omegaconf.dictconfig):
    logger = logging.getLogger(__name__)
    logger.info("Setting up logging configuration\n")
    setup_logging(
        logging_config_path=os.path.join(
            hydra.utils.get_original_cwd(), "conf", "logging.yaml"
        )
    )

    evaluation_pipeline = EvaluationPipeline(cfg=cfg, logger=logger)
    evaluation_pipeline.evaluation()


if __name__ == "__main__":
    main()
