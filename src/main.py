import logging
import os

import hydra
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores.faiss import FAISS
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

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg["embeddings"]["chunk_size"],
        chunk_overlap=cfg["embeddings"]["chunk_overlap"],
    )
    texts = text_splitter.split_documents(documents)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        embeddings = HuggingFaceInstructEmbeddings(
            model_name=cfg["embeddings"]["embeddings_model"],
            model_kwargs={"device": device},
        )
        logger.info(f"Embedding Model loaded to {device.upper()}.")

    except RuntimeError as e:
        logger.error(f"CUDA memory issue: {e}. Switching to CPU.")
        embeddings = HuggingFaceInstructEmbeddings(
            model_name=cfg["embeddings"]["embeddings_model"],
            model_kwargs={"device": "cpu"},
        )

    if not os.path.exists(cfg["embeddings"]["embeddings_path"]):
        os.makedirs(name=cfg["embeddings"]["embeddings_path"], exist_ok=True)

    logger.info("Generating Vector Embeddings")
    vectordb = FAISS.from_documents(documents=texts, embedding=embeddings)
    logger.info("Saving Vector Embeddings")
    vectordb.save_local(
        folder_path=cfg["embeddings"]["embeddings_path"],
        index_name=cfg["embeddings"]["index_name"],
    )

    logging.info("Loading Vector DB")
    vectordb = FAISS.load_local(
        folder_path=cfg["embeddings"]["embeddings_path"],
        index_name=cfg["embeddings"]["index_name"],
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )
    logging.info("Successfully loaded")


if __name__ == "__main__":
    main()
