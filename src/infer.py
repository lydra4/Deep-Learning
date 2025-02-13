import logging
import os

import hydra
from dotenv import load_dotenv
from langchain.chains.llm import LLMChain
from langchain_core.prompts.prompt import PromptTemplate
from langchain_openai.chat_models import ChatOpenAI
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

    if not os.path.exists(cfg["embeddings_path"]):
        raise FileNotFoundError(f"The path, {cfg['embeddings_path']}, does not exits.")

    logger.info("Loading Vector DB")

    folder_path = os.path.dirname(cfg["embeddings_path"])
    index_name = os.path.basename(cfg["embeddings_path"])

    # vectordb = FAISS.load_local(folder_path=folder_path, index_name=index_name)

    logger.info("Successfully loaded")

    template = """
    If you do not know, do not make up an answer, mention that you do not know.
    Answer in the same language as the question.

    {context_clause}

    Question: {question}
    Answer:"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["question"],
        partial_variables={
            "context_clause": "Use the follow context to answer:\n{context}"
        },
    )

    load_dotenv()

    llm = ChatOpenAI(
        model="gpt-4",
        temperature=0.7,
        api_key=os.getenv("api_key"),
    )

    llm_chain = LLMChain(prompt=prompt, llm=llm)


if __name__ == "__main__":
    main()
