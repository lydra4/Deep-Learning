import logging
import os
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

    def _split_text(self) -> List[Document]:
        """Splits documents into smaller chunks based on configuration.

        Returns:
            List[Document]: List of split text documents.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.cfg["embeddings"]["chunk_size"],
            chunk_overlap=self.cfg["embeddings"]["chunk_overlap"],
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

    def _embed_documents(self) -> FAISS:
        """Splits, embeds documents, and saves the vector store to disk.

        Returns:
            FAISS: The FAISS vector store containing the document embeddings.
        """
        if not self.texts:
            self._split_text()

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
        return vectordb

    def load_vectordb(self) -> FAISS:
        """Loads a saved FAISS vector database.

        Returns:
            FAISS: The loaded FAISS vector store.

        Raises:
            FileNotFoundError: If the FAISS vector database is not found.
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

    def process_and_load_vectordb(self) -> FAISS:
        """Processes documents, generates embeddings, and loads FAISS vector DB.

        Returns:
            FAISS: The loaded FAISS vector store.
        """
        if self.logger:
            self.logger.info("Starting document processing and generating embeddings")
        self._split_text()
        self._embed_documents()
        return self.load_vectordb()
