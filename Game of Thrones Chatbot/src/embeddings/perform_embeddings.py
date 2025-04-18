import logging
import os
import re
from typing import List, Optional, Tuple

import clip
import faiss
import numpy as np
import omegaconf
import torch
from langchain.docstore.document import Document
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_experimental.text_splitter import SemanticChunker
from PIL import Image


class PerformEmbeddings:
    """
    Handles document embedding and FAISS vector database operations.

    Attributes:
        cfg (omegaconf.DictConfig): Configuration settings.
        documents (List[Document]): List of documents to process.
        logger (logging.Logger): Logger instance for logging messages.
        texts (List[Document]): List of processed text documents.
        embedding_model (Optional[HuggingFaceInstructEmbeddings]): Loaded embedding model.
        embeddings_path (Optional[str]): Path where embeddings are stored.
        embeddings_model_name (Optional[str]): Name of the embedding model.
    """

    def __init__(
        self,
        cfg: omegaconf.DictConfig,
        documents: List[Document],
        images: Optional[List[Tuple[str, Image.Image]]],
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """
        Initializes the PerformEmbeddings class.

        Args:
            cfg (omegaconf.DictConfig): Configuration dictionary.
            documents (List[Document]): List of documents to be processed.
            logger (Optional[logging.Logger], optional): Logger instance. Defaults to None.
        """
        self.cfg = cfg
        self.logger = logger or logging.getLogger(__name__)
        self.documents = documents
        self.images = images
        self.texts: List[Document] = []
        self.embedding_model: Optional[HuggingFaceInstructEmbeddings] = None
        self.embeddings_path: Optional[str] = None
        self.embeddings_model_name: Optional[str] = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _split_text(self) -> List[Document]:
        """
        Splits the input documents into smaller chunks based on the configured text splitter.

        Returns:
            List[Document]: A list of split text documents.
        """
        self.embedding_model = self._load_embeddings_model()

        self.logger.info(f"Using {self.cfg.text_splitter.name.replace('_', ' ')}.\n")

        if self.cfg.text_splitter.name.lower() == "recursive_character_text_splitter":
            text_splitter = RecursiveCharacterTextSplitter(
                **self.cfg.text_splitter.text_splitter
            )

        elif (
            self.cfg.text_splitter.name.lower()
            == "sentence_transformers_token_text_splitter"
        ):
            text_splitter = SentenceTransformersTokenTextSplitter(
                model_name=self.cfg.embeddings.embeddings_model.model_name,
                chunk_overlap=self.cfg.text_splitter.text_splitter.chunk_overlap,
            )

        elif self.cfg.text_splitter.name.lower() == "semantic_chunker":
            text_splitter = SemanticChunker(
                embeddings=self.embedding_model,
                breakpoint_threshold_type=self.cfg.text_splitter.text_splitter.breakpoint_threshold_type,
            )

        self.texts = text_splitter.split_documents(self.documents)
        self.logger.info(f"Text split into {len(self.texts)} parts.")

        return self.texts

    def _load_embeddings_model(self) -> HuggingFaceInstructEmbeddings:
        """
        Loads the HuggingFace embedding model onto the appropriate device (CPU or GPU).

        Returns:
            HuggingFaceInstructEmbeddings: The loaded embedding model.
        """
        self.logger.info(f"Embedding model will be loaded to {self.device}.\n")

        model_config = {
            "model_name": self.cfg.embeddings.embeddings_model.model_name,
            "show_progress": self.cfg.embeddings.embeddings_model.show_progress,
            "model_kwargs": {"device": self.device},
        }
        self.embedding_model = HuggingFaceInstructEmbeddings(**model_config)

        self.logger.info(f"Embedding Model loaded to {self.device.upper()}.\n")

        return self.embedding_model

    def _normalize_embeddings(self, embeddings):
        return embeddings / np.linalg.norm(x=embeddings, axis=1, keepdims=True)

    def _embed_documents(self) -> FAISS:
        """
        Splits the documents, generates embeddings, and saves the vector store.

        Returns:
            FAISS: The FAISS vector store containing the document embeddings.
        """
        if not self.texts:
            self._split_text()

        self.embeddings_model_name = re.sub(
            r'[<>:"/\\|?*]',
            "_",
            self.cfg.embeddings.embeddings_model.model_name.split("/")[-1],
        )

        self.embeddings_path = os.path.join(
            self.cfg.embeddings.embed_documents.embeddings_path,
            self.cfg.text_splitter.name,
            self.embeddings_model_name,
        )
        os.makedirs(self.embeddings_path, exist_ok=True)
        self.logger.info(f"Embeddings will be saved @ {self.embeddings_path}\n")

        self.logger.info("Generating Vector Embeddings.\n")

        vectordb = self._normalize_embeddings(
            FAISS.from_documents(documents=self.texts, embedding=self.embedding_model)
        )

        self.logger.info("Saving Vector Embeddings.\n")

        vectordb.save_local(
            folder_path=self.embeddings_path,
            index_name=self.cfg.embeddings.embed_documents.index_name,
        )

        self.logger.info("Successfully saved.\n")

    def _embed_images(self) -> None:
        self.logger.info(f"CLIP model will be loaded to {self.device}.\n")

        model, preprocess = clip.load(
            name=self.cfg.embeddings.embeddings_model.clip_model_name,
            device=self.device,
        )

        documents = []
        embeddings = []

        for name, image in self.images:
            try:
                preprocessed_image = preprocess(img=image).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    embedding = model.encode_image(preprocessed_image)
                embedding = embedding.cpu().squeeze(0).numpy()

                documents.append(
                    Document(page_content="", metadata={"image_name": name})
                )
                embeddings.append(embedding)

            except Exception as e:
                self.logger.warning(f"Failed to embed image {image}: {e}.\n")

        if not embeddings:
            self.logger.warning("No image embeddings were generated.")

        clip_model_name_clean = re.sub(
            r'[<>:"/\\|?*]', "_", self.cfg.embeddings.embeddings_model.clip_model_name
        )
        image_embeddings_path = os.path.join(
            self.cfg.embeddings.embed_images.embeddings_path, clip_model_name_clean
        )

        os.makedirs(image_embeddings_path, exist_ok=True)

        self.logger.info(f"Saving image embeddings to {image_embeddings_path}.\n")

        embedding_matrix = self._normalize_embeddings(
            np.array(embeddings).astype("float32")
        )
        index = faiss.IndexHNSWFlat(embedding_matrix.shape[1], 32)
        index.add(embedding_matrix)

    def generate_vectordb(self) -> FAISS:
        """
        Processes the documents, generates embeddings, and loads the FAISS vector database.

        Returns:
            FAISS: The FAISS vector store containing the embeddings.
        """
        self.logger.info("Starting document processing and generating embeddings.\n")
        self._split_text()
        self._embed_documents()

        if self.images:
            self._embed_images()
