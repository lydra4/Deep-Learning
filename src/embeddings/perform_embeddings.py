import logging
import os
import re
from typing import List, Optional

import torch
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores.faiss import FAISS


class PerformEmbeddings:
    """Handles document embedding and FAISS vector database operations."""

    def __init__(
        self,
        cfg: dict,
        documents: List[Document],
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """Initializes the PerformEmbeddings class.

        Args:
            cfg (dict): Configuration dictionary.
            documents (List[Document]): List of documents to process.
            logger (Optional[logging.Logger]): Logger instance for logging messages.
        """
        self.cfg = cfg
        self.logger = logger
        self.documents = documents
        self.texts: List[Document] = []
        self.embeddings: Optional[HuggingFaceInstructEmbeddings] = None
        self.embeddings_path: Optional[str] = None

    def _split_text(self) -> List[Document]:
        """Splits documents into smaller chunks based on configuration.

        Returns:
            List[Document]: List of split text documents.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            **self.cfg["embeddings"]["text_splitter"],
        )
        self.texts = text_splitter.split_documents(self.documents)
        if self.logger:
            self.logger.info(f"Text split into {len(self.texts)} parts.")
        return self.texts

    def _load_embeddings(self, device: str) -> HuggingFaceInstructEmbeddings:
        """Loads the HuggingFace embedding model on the specified device.

        Args:
            device (str): The device to load the embeddings model on.

        Returns:
            HuggingFaceInstructEmbeddings: The loaded embeddings model.
        """
        model_config = {
            **self.cfg["embeddings"]["load_embeddings"],
            "model_kwargs": {"device": device},
        }
        embeddings = HuggingFaceInstructEmbeddings(**model_config)
        if self.logger:
            self.logger.info(f"Embedding Model loaded to {device.upper()}")

        return embeddings

    def _embed_documents(self) -> FAISS:
        """Splits, embeds documents, and saves the vector store to disk.

        Returns:
            FAISS: The FAISS vector store containing the document embeddings.
        """
        if not self.texts:
            self._split_text()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.logger:
            self.logger.info(f"Embedding will be loaded to {device}")
        self.embeddings = self._load_embeddings(device=device)

        embeddings_model_name = re.sub(
            r'[<>:"/\\|?*]',
            "_",
            self.cfg["embeddings"]["load_embeddings"]["model_name"].split("/")[-1],
        )
        index_name = re.sub(
            r'[<>:"/\\|?*]',
            "_",
            self.cfg["embeddings"]["embed_documents"]["index_name"],
        )

        self.embeddings_path = os.path.join(
            self.cfg["embeddings"]["embed_documents"]["embeddings_path"],
            embeddings_model_name,
            index_name,
        )
        if not os.path.exists(self.embeddings_path):
            os.makedirs(self.embeddings_path, exist_ok=True)
            if self.logger:
                self.logger.info(f"Created folder at {self.embeddings_path}")
        else:
            if self.logger:
                self.logger.info(f"Folder already exits at {self.embeddings_path}")

        if self.logger:
            self.logger.info("Generating Vector Embeddings")

        vectordb = FAISS.from_documents(documents=self.texts, embedding=self.embeddings)

        if self.logger:
            self.logger.info("Saving Vector Embeddings")

        vectordb.save_local(
            folder_path=self.embeddings_path,
            index_name=self.cfg["embeddings"]["embed_documents"]["index_name"],
        )

        if self.logger:
            self.logger.info("Successfully saved.")
        return vectordb

    def generate_vectordb(self) -> FAISS:
        """Processes documents, generates embeddings, and loads FAISS vector DB.

        Returns:
            FAISS: The loaded FAISS vector store.
        """
        if self.logger:
            self.logger.info("Starting document processing and generating embeddings")
        self._split_text()
        self._embed_documents()
