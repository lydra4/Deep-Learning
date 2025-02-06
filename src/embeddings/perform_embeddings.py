import logging
import os
from typing import List, Optional

import torch
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores.faiss import FAISS


class PerformEmbeddings:
    """A class to handle document splitting, embedding generation, and vector database operations.

    Attributes:
        cfg (dict): Configuration dictionary containing settings for embeddings and vector database.
        documents (List[Document]): List of documents to be processed.
        logger (Optional[logging.Logger]): Logger instance for logging messages.
        texts (List[Document]): List of split documents.
        embeddings (HuggingFaceInstructEmbeddings): Embeddings model instance.
    """

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

    def split_text(self) -> List[Document]:
        """Splits the documents into smaller chunks using a text splitter.

        Returns:
            List[Document]: List of split documents.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.cfg["embeddings"]["chunk_size"],
            chunk_overlap=self.cfg["embeddings"]["chunk_overlap"],
        )
        self.texts = text_splitter.split_documents(self.documents)
        return self.texts

    def _load_embeddings(self, device: str) -> HuggingFaceInstructEmbeddings:
        """Loads the embeddings model on the specified device.

        Args:
            device (str): Device to load the model on ('cuda' or 'cpu').

        Returns:
            HuggingFaceInstructEmbeddings: Loaded embeddings model.

        Raises:
            RuntimeError: If the model fails to load on the specified device.
        """
        try:
            embeddings = HuggingFaceInstructEmbeddings(
                model_name=self.cfg["embeddings"]["embeddings_model"],
                model_kwargs={"device": device},
            )
            if self.logger:
                self.logger.info(f"Embedding Model loaded to {device.upper()}")
            return embeddings

        except RuntimeError as e:
            if self.logger:
                self.logger.error(f"CUDA memory issue: {e}. Switching to CPU")
            return HuggingFaceInstructEmbeddings(
                model_name=self.cfg["embeddings"]["embeddings_model"],
                multi_process=self.cfg["embeddings"]["multiprocessing"],
                model_kwargs={"device": "cpu"},
            )

    def embed_documents(self):
        """Generates embeddings for the split documents and saves them to a vector database.

        Raises:
            ValueError: If no text data is available for embedding.
        """
        if not self.texts:
            raise ValueError("No text data available.")

        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.embeddings = self._load_embeddings(device=device)

        embeddings_path = self.cfg["embeddings"]["embeddings_path"]
        os.makedirs(embeddings_path, exist_ok=True)

        if self.logger:
            self.logger.info("Generating Vector Embeddings")

        vectordb = FAISS.from_documents(documents=self.texts, embedding=self.embeddings)

        if self.logger:
            self.logger.info("Saving Vector Embeddings")

        vectordb.save_local(
            folder_path=embeddings_path, index_name=self.cfg["embeddings"]["index_name"]
        )

        if self.logger:
            self.logger.info("Successfully saved.")

    def load_vectordb(self) -> FAISS:
        """Loads the vector database from the specified path.

        Returns:
            FAISS: Loaded vector database.

        Raises:
            FileNotFoundError: If the vector database files do not exist.
        """
        vectordb_path = os.path.join(
            self.cfg["embeddings"]["embeddings_path"],
            self.cfg["embeddings"]["index_name"],
        )

        if not os.path.exists(vectordb_path):
            raise FileNotFoundError(
                f"{self.cfg['embeddings']['index_name']} does not exits at {self.cfg['embeddings']['embeddings_path']}. Please check."
            )

        if self.logger:
            self.logger.info("Loading Vector DB")

        vectordb = FAISS.load_local(
            folder_path=self.cfg["embeddings"]["embeddings_path"],
            index_name=self.cfg["embeddings"]["index_name"],
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True,
        )

        if self.logger:
            self.logger.info("Successfully loaded")

        return vectordb
