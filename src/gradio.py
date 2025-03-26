import logging
from typing import Optional

import omegaconf
from src.inference.inference_pipeline import InferencePipeline


class GradioApp:
    def __init__(
        self, cfg: omegaconf.DictConfig, logger: Optional[logging.Logger] = None
    ) -> None:
        self.cfg = cfg
        self.logger = logger or logging.getLogger(__name__)
        self.inference_pipeline = InferencePipeline(cfg=cfg, logger=self.logger)

        self.inference_pipeline.load_embedding_model()
        self.inference_pipeline._load_vectordb()
        self.inference_pipeline._initialize_llm()
        self.inference_pipeline._create_retriever()
        self.inference_pipeline._create_qa_chain()
