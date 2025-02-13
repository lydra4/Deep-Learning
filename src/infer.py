import logging
import os

import hydra
import torch
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores.faiss import FAISS
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

    index_name = os.path.basename(cfg["embeddings_path"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Embedding Model will be loaded to {device.upper()}.")

    model_config = {
        "model_name": cfg["embeddings_model_name"],
        "show_progress": cfg["show_progress"],
        "model_kwargs": {"device": device},
    }

    embedding_model = HuggingFaceInstructEmbeddings(**model_config)

    logger.info(f"Embedding Model loaded to {device.upper()}.")

    vectordb = FAISS.load_local(
        folder_path=cfg["embeddings_path"],
        embeddings=embedding_model,
        index_name=index_name,
        allow_dangerous_deserialization=True,
    )

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

    llm_chain = prompt | llm

    response = llm_chain.invoke


if __name__ == "__main__":
    main()
