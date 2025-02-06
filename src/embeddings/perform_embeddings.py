import logging
import os
from typing import List, Optional

import torch
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores.faiss import FAISS


class PerformEmbeddings:
    def __init__(
        self,
        cfg: dict,
        documents: List[Document],
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.cfg = cfg
        self.logger = logger
        self.documents = documents
        self.texts: List[Document] = []

    def split_text(self) -> List[Document]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.cfg["embeddings"]["chunk_size"],
            chunk_overlap=self.cfg["embeddings"]["chunk_overlap"],
        )

        self.texts = text_splitter.split_documents(self.documents)

        return self.texts

    def embed_documents(self):
        if not self.texts:
            raise ValueError("No text data available")

        device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            embeddings = HuggingFaceInstructEmbeddings(
                model_name=self.cfg["embeddings"]["embeddings_model"],
                model_kwargs={"device": device},
            )
            if self.logger:
                self.logger.info(f"Embedding Model loaded to {device.upper()}")

        except RuntimeError as e:
            if self.logger:
                self.logger.error(f"CUDA memory issue: {e}. Switching to CPU")
            embeddings = HuggingFaceInstructEmbeddings(
                model_name=self.cfg["embeddings"]["embeddings_model"],
                multi_process=self.cfg["embeddings"]["multiprocessing"],
                model_kwargs={"device": "cpu"},
            )

        if not os.path.exists(self.cfg["embeddings"]["embeddings_path"]):
            os.makedirs(name=self.cfg["embeddings"]["embeddings_path"], exist_ok=True)

        if self.logger:
            self.logger.info("Generating Vector Embeddings")

        vectordb = FAISS.from_documents(documents=self.texts, embedding=embeddings)

        if self.logger:
            self.logger.info("Saving Vector Embeddings")

        vectordb.save_local(
            folder_path=self.cfg["embeddings"]["embeddings_path"],
            index_name=self.cfg["embeddings"]["index_name"],
        )
