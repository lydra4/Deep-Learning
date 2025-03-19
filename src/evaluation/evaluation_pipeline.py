import json
import logging
import os
from datetime import datetime
from typing import Optional

import omegaconf
import pandas as pd
import pytz
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
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

from inference.inference_pipeline import InferencePipeline


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
        self.no_of_questions: Optional[int] = None

    def _run_inference(self):
        """
        Runs inference using the InferencePipeline.

        Raises:
            RuntimeError: If inference fails.
        """
        try:
            infer_pipeline = InferencePipeline(cfg=self.cfg, logger=self.logger)
            self.embedding_model = infer_pipeline.load_embedding_model()
            self.ragas_df, self.no_of_questions = infer_pipeline.run_inference()
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
        model = self.cfg.model
        api_key_env_var = "OPENAI_API_KEY" if "gpt-" in model else "GEMINI_API_KEY"
        api_key = os.getenv(api_key_env_var)

        if not api_key:
            self.logger.error(
                "API key not found. Ensure you have a .env file with `api_key`"
            )
            raise ValueError("API key not found in environment variables.")

        try:
            if "gpt-" in model:
                llm = ChatOpenAI(model=model, api_key=api_key)

            elif "gemini" in model:
                llm = ChatGoogleGenerativeAI(model=model, api_key=api_key)

            else:
                raise ValueError(f"Unsupported model: {model}.")

            self.evaluator_llm = LangchainLLMWrapper(langchain_llm=llm)
            self.logger.info(f"{model} successfully initialized.\n")

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
        self._run_inference()
        self._initialize_llm()
        self._setup_metrics()

        try:
            self.logger.info("Evaluating results.\n")
            results = evaluate(dataset=self.ragas_df, metrics=self.metrics)
            self.logger.info(f"Results: {results}")

            sgt_timezone = pytz.timezone(self.cfg.timezone)
            timestamp_sgt = datetime.now(sgt_timezone).strftime(
                self.cfg.datetime_format
            )

            results_df = results.to_pandas()
            metrics_list = results_df.columns[-len(self.metrics) :]
            results_dict = results_df[metrics_list].mean().to_dict()

            evaluation_entry = {
                "timestamp": timestamp_sgt,
                "no_of_questions": self.no_of_questions,
                "embedding_model": self.cfg.embeddings.embeddings_model.model_name,
                "llm": self.cfg.model,
                "metrics": results_dict,
            }

            evaluation_log = os.path.join(self.cfg.metric_path, "evaluation_log.json")
            if os.path.exists(evaluation_log):
                with open(evaluation_log, "r", encoding="utf-8") as f:
                    try:
                        log_data = json.load(f)
                        if not isinstance(log_data, list):
                            log_data = []
                    except json.JSONDecodeError:
                        log_data = []

            else:
                log_data = []

            log_data.append(evaluation_entry)

            with open(evaluation_log, "w", encoding="utf-8") as f:
                json.dump(log_data, f, indent=4)
                self.logger.info(f"Metrics saved at {evaluation_log}")

        except Exception as e:
            self.logger.error(f"Evaluation failed: {e}")
            raise RuntimeError(
                "Evaluation process failed. Check Configurations."
            ) from e
