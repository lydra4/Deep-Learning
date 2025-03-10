import logging
import os
from typing import Optional

import omegaconf
import pandas as pd
import torch
from dotenv import load_dotenv
from inference.inference_pipeline import InferencePipeline
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    AnswerRelevancy,
    ContextPrecision,
    FactualCorrectness,
    Faithfulness,
    LLMContextRecall,
)


class EvaluationPipeline:
    """
    A pipeline for evaluating a retrieval-augmented generation (RAG) system using various metrics.

    Attributes:
        cfg (omegaconf.DictConfig): The configuration object.
        logger (logging.Logger): Logger instance for logging messages.
        embedding_model (Optional[HuggingFaceInstructEmbeddings]): Embedding model used for retrieval evaluation.
        ragas_df (Optional[pd.DataFrame]): Dataframe storing inference results.
        evaluator_llm (Optional[LangchainLLMWrapper]): LLM wrapper for evaluation.
        metrics (List[object]): List of evaluation metrics used in the pipeline.
    """

    def __init__(
        self, cfg: omegaconf.DictConfig, logger: Optional[logging.Logger] = None
    ) -> None:
        """
        Initializes the EvaluationPipeline with configuration and logging.

        Args:
            cfg (omegaconf.DictConfig): Configuration object.
            logger (Optional[logging.Logger]): Logger instance. Defaults to None.
        """
        self.cfg = cfg
        self.logger = logger or logging.getLogger(__name__)
        self.embedding_model: Optional[HuggingFaceInstructEmbeddings] = None
        self.ragas_df: Optional[pd.DataFrame] = None
        self.evaluator_llm: Optional[LangchainLLMWrapper] = None
        self.metrics: list[object] = []

    def _load_embedding_model(self):
        """
        Loads the embedding model based on the configuration settings.

        Raises:
            RuntimeError: If the embedding model fails to load.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            self.embedding_model = HuggingFaceInstructEmbeddings(
                model_name=self.cfg.embeddings.embeddings_model.model_name,
                show_progress=self.cfg.show_progress,
                model_kwargs={"device": device},
            )
            self.logger.info(
                f"{self.cfg.embeddings.embeddings_model.model_name} loaded to {device.upper()}"
            )
        except Exception as e:
            self.logger.error(f"Failed to load embedding model: {e}")
            raise RuntimeError(
                "Embedding model failed to load. Please check configuration."
            ) from e

    def _run_inference(self):
        """
        Runs inference using the InferencePipeline.

        Raises:
            RuntimeError: If inference fails.
        """
        try:
            infer_pipeline = InferencePipeline(cfg=self.cfg, logger=self.logger)
            self.ragas_df = infer_pipeline.run_inference()
        except Exception as e:
            self.logger.error(f"Failed to run inference: {e}")
            raise RuntimeError(
                "Inference failed. Check the pipeline configuration."
            ) from e

    def _initialize_llm(self):
        """
        Initializes the LLM model for evaluation.

        Raises:
            ValueError: If the API key is missing.
            RuntimeError: If LLM initialization fails.
        """
        load_dotenv()
        api_key = os.getenv("api_key")

        if not api_key:
            self.logger.error(
                "API key not found. Ensure you have a .env file with `api_key`"
            )
            raise ValueError("API key not found in environment variables.")

        try:
            self.evaluator_llm = LangchainLLMWrapper(
                ChatOpenAI(model=self.cfg.model, api_key=api_key)
            )
            self.logger.info(f"{self.cfg.model} successfully initialized.")
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM: {e}")
            raise RuntimeError(
                "LLM initializaion failed. Check API key and model name"
            ) from e

    def _setup_metrics(self):
        """
        Sets up evaluation metrics based on the configuration.

        Raises:
            RuntimeError: If LLM or embedding model is not initialized.
        """
        if self.evaluator_llm is None:
            raise RuntimeError("LLM is not initialized. Call _initialize_llm first")

        if self.embedding_model is None:
            raise RuntimeError(
                "Embedding model is not initialized. Call _load_embedding_model first"
            )

        metric_mapping = {
            "answer_relevancy": AnswerRelevancy(
                llm=self.evaluator_llm, embeddings=self.embedding_model
            ),
            "context_precision": ContextPrecision(llm=self.evaluator_llm),
            "llm_context_recall": LLMContextRecall(llm=self.evaluator_llm),
            "faithfulness": Faithfulness(llm=self.evaluator_llm),
            "factual_correctness": FactualCorrectness(llm=self.evaluator_llm),
        }
        self.metrics = [
            metric
            for key, metric in metric_mapping.items()
            if getattr(self.cfg.metrics, key, False)
        ]

    def evaluation(self):
        """
        Executes the full evaluation pipeline, including:
        1. Loading the embedding model.
        2. Running inference.
        3. Initializing the LLM.
        4. Setting up evaluation metrics.
        5. Running the evaluation process.

        Raises:
            RuntimeError: If any component fails during execution.
        """
        self._load_embedding_model()
        self._run_inference()
        self._initialize_llm()
        self._setup_metrics()

        try:
            results = evaluate(dataset=self.ragas_df, metrics=self.metrics)
            self.logger.info(f"Results: {results}")
        except Exception as e:
            self.logger.error(f"Evaluation failed: {e}")
            raise RuntimeError(
                "Evaluation process failed. Check Configurations."
            ) from e
