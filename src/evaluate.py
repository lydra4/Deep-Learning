import logging
import os

import hydra
import omegaconf
from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    AnswerRelevancy,
    ContextPrecision,
)

from inference.inference_pipeline import InferencePipeline
from utils.general_utils import setup_logging


@hydra.main(version_base=None, config_path="../conf", config_name="evaluate.yaml")
def main(cfg: omegaconf.dictconfig):
    logger = logging.getLogger(__name__)
    logger.info("Setting up logging configuration")
    setup_logging(
        logging_config_path=os.path.join(
            hydra.utils.get_original_cwd(), "conf", "logging.yaml"
        )
    )
    load_dotenv()
    api_key = os.getenv("api_key")
    if not api_key:
        raise ValueError("API key not found in environment variables.")

    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o", api_key=api_key))

    infer_pipeline = InferencePipeline(cfg=cfg, logger=logger)
    ragas_df = infer_pipeline.run_inference()

    metrics = [
        AnswerRelevancy(llm=evaluator_llm),
        ContextPrecision(llm=evaluator_llm),
        ContextPrecision(llm=evaluator_llm),
    ]

    results = evaluate(dataset=ragas_df, metrics=metrics)

    logger.info(f"Results: {results}")


if __name__ == "__main__":
    main()
