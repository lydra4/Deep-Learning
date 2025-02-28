import locale
import logging
import os
from typing import Optional

import omegaconf
import torch
from dotenv import load_dotenv
from langchain.prompts.prompt import PromptTemplate
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_openai.chat_models import ChatOpenAI


class InferencePipeline:
    def __init__(
        self, cfg: omegaconf.DictConfig, logger: Optional[logging.Logger] = None
    ) -> None:
        self.cfg = cfg
        self.logger = logger or logging.getLogger(__name__)
        self.embedding_model: Optional[HuggingFaceInstructEmbeddings] = None
        self.vectordb: Optional[FAISS] = None
        self.prompt: Optional[PromptTemplate] = None
        self.llm: Optional[ChatOpenAI] = None

    def _load_embedding_model(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"Embedding Model will be loaded to {device.upper()}\n")

        self.logger.info("Loading embedding model")
        model_config = {
            "model_name": self.cfg.embeddings.embeddings_model_name,
            "show_progress": self.cfg.embeddings.show_progress,
            "model_kwargs": {"device": device},
        }
        self.embedding_model = HuggingFaceInstructEmbeddings(**model_config)
        self.logger.info(f"Embedding Model loaded to {device.upper()}\n")

    def _load_vectordb(self):
        if not os.path.exists(self.cfg.embeddings.embeddings_path):
            raise FileNotFoundError(
                f"The path, {self.cfg.embeddings.embeddings_path}, does not exits"
            )

        self._load_embedding_model()
        index_name = os.path.basename(self.cfg.embeddings.embeddings_path)
        self.logger.info("Loading Vector database")

        self.vectordb = FAISS.load_local(
            folder_path=self.cfg.embeddings.embeddings_path,
            embeddings=self.embedding_model,
            index_name=index_name,
            allow_dangerous_deserialization=True,
        )

        self.logger.info("Successfully Loaded")

    def _prompt_template(self):
        self.logger.info("Loading Prompt Template")
        with open(
            file=self.cfg.path_to_template, mode="r", encoding=locale.getencoding()
        ) as f:
            template = f.read()

        self.prompt = PromptTemplate(
            template=template, input_variables=["context", "question"]
        )

    def _intialize_llm(self):
        load_dotenv(dotenv_path="../.env")
        self.llm = ChatOpenAI(
            model=self.cfg.llm.model,
            temperature=self.cfg.llm.temperature,
            api_key=os.getenv("api_key"),
        )
