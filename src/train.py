import logging
import os

import hydra
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from utils.general_utils import setup_logging


@hydra.main(version_base=None, config_path="../conf", config_name="training.yaml")
def main(cfg):
    logger = logging.getLogger(__name__)
    logger.info("Setting up logging configuration.")
    setup_logging(
        logging_config_path=os.path.join(
            hydra.utils.get_original_cwd(), "conf", "logging.yaml"
        )
    )

    text_loader = DirectoryLoader(
        path=cfg["preprocessing"]["path"],
        glob=cfg["preprocessing"]["glob"],
        show_progress=cfg["preprocessing"]["show_progress"],
        use_multithreading=cfg["preprocessing"]["use_multithreading"],
        sample_seed=cfg["preprocessing"]["sample_seed"],
    )

    try:
        logger.info(f"Loading {cfg['preprocessing']['glob']}.")
        documents = text_loader.load()
        if not documents:
            logger.warning("No documents were loaded. Please check the directory path or glob pattern.")
            return
        logger.info(f"{cfg['preprocessing']['glob']} successfully loaded.")

    except Exception as e:
        logger.error(f"Error loading documents, {e}", exc_info=True)
        return

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg["preprocessing"]["chunk_size"],
        chunk_overlap=cfg["preprocessing"]["chunk_overlap"],
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

    if not os.path.exists(cfg["embeddings"]["embeddings_path"])
        os.makedirs(name=cfg["embeddings"]["embeddings_path"], exist_ok=True)

    logger.info("Generating Vector Embeddings")
    vectordb = FAISS.from_documents(documents=texts, embedding=embeddings)
    logger.info("Saving Vector Embeddings")
    vectordb.save_local(
        folder_path=cfg["embeddings"]["embeddings_path"], index_name="faiss_index_got"
    )


if __name__ == "__main__":
    main()
