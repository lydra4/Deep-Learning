import logging
import os

import hydra
import torch
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader
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

    logger.info(f"Loading {cfg['preprocessing']['glob']}.")
    documents = text_loader.load()
    logger.info(f"{cfg['preprocessing']['glob']} successfully loaded.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg["preprocessing"]["chunk_size"],
        chunk_overlap=cfg["preprocessing"]["chunk_overlap"],
    )
    texts = text_splitter.split_documents(documents=)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Embedding Model is loaded to {device}.")
    embeddings = HuggingFaceInstructEmbeddings(
        model_name=cfg["embeddings"]["embeddings_model"],
        model_kwargs={"device": device},
    )

    vectordb = FAISS.from_documents(
        documents=documents,
        embedding=embeddings
    )


if __name__ == "__main__":
    main()
