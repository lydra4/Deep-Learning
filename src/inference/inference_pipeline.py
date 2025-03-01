import locale
import logging
import os
from typing import Optional

import omegaconf
import pandas as pd
import torch
from datasets import Dataset
from dotenv import load_dotenv
from langchain.chains.retrieval_qa.base import RetrievalQA
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
        self.retriever = None
        self.qa_chain = None
        self.qns_list: Optional[list] = None
        self.answer_file: Optional[str] = None

    def _load_embedding_model(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"Loading embedding model on {device.upper()}\n")

        try:
            self.embedding_model = HuggingFaceInstructEmbeddings(
                model_name=self.cfg.embeddings.embeddings_model_name,
                show_progress=self.cfg.embeddings.show_progress,
                model_kwargs={"device": device},
            )
            self.logger.info("Embedding model loaded successfully.")

        except Exception as e:
            self.logger.error(f"Failed to load embedding model: {e}")
            raise

    def _load_vectordb(self):
        if not os.path.exists(self.cfg.embeddings.embeddings_path):
            raise FileNotFoundError(
                f"Vector database path does not exist: {self.cfg.embeddings.embeddings_path}"
            )

        index_name = os.path.basename(self.cfg.embeddings.embeddings_path)
        self.logger.info("Loading Vector database")

        try:
            self.vectordb = FAISS.load_local(
                folder_path=self.cfg.embeddings.embeddings_path,
                embeddings=self.embedding_model,
                index_name=index_name,
                allow_dangerous_deserialization=True,
            )
            self.logger.info("Vector database loaded successfully.")

        except Exception as e:
            self.logger.error(f"Failed to load vector database: {e}")
            raise

    def _load_prompt_template(self):
        self.logger.info("Loading Prompt Template")

        try:
            with open(
                file=self.cfg.path_to_template, mode="r", encoding=locale.getencoding()
            ) as f:
                template = f.read()

            self.prompt = PromptTemplate(
                template=template, input_variables=["context", "question"]
            )
            self.logger.info("Prompt template loaded successfully.")

        except Exception as e:
            self.logger.error(f"Failed to load Prompt Template: {e}")
            raise

    def _initialize_llm(self):
        load_dotenv()
        api_key = os.getenv("api_key")
        if not api_key:
            raise ValueError("API key not found in environment variables.")

        self.logger.info("Initializing LLM")
        try:
            self.llm = ChatOpenAI(
                model=self.cfg.llm.model,
                temperature=self.cfg.llm.temperature,
                api_key=api_key,
            )
            self.logger.info(
                f"LLM successfully initialized with model: {self.cfg.llm.model}"
            )

        except Exception as e:
            self.logger.error(f"Failed to initialize LLM: {e}")
            raise

    def _create_retriever(self):
        self.retriever = self.vectordb.as_retriever(
            search_kwargs={
                "k": self.cfg.retrieve.k,
                "search_type": self.cfg.retrieve.search_type,
            }
        )

    def _create_qa_chain(self):
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type=self.cfg.retrieval.chain_type,
            retriever=self.retriever,
            chain_type_kwargs={"prompt": self.prompt},
            return_source_documents=self.cfg.retrieval.return_source_documents,
            verbose=self.cfg.retrieval.verbose,
        )

    def _open_questions(self):
        self.logger.info("Loading questions...")

        try:
            with open(
                file=self.cfg.llm.path_to_qns, mode="r", encoding=locale.getencoding()
            ) as f:
                self.qns_list = [line.rstrip("\n").strip() for line in f.readlines()]

            self.logger.info(f"Loaded {len(self.qns_list)} questions.")

        except Exception as e:
            self.logger.error(f"Failed to load questions: {e}")
            raise

    def _infer(self):
        folder_to_answers = os.path.dirname(self.cfg.llm.path_to_ans)
        os.makedirs(name=folder_to_answers, exist_ok=True)

        self.logger.info(f"Saving answers to {self.cfg.llm.path_to_ans}")
        data_list = []

        with open(
            file=self.cfg.llm.path_to_ans,
            mode="w",
            encoding=locale.getencoding(),
            newline="\n",
        ) as self.answer_file:
            for question in self.qns_list:
                retrieved_docs = self.retriever.invoke(input=question)

                for document in retrieved_docs:
                    document.page_content = document.page_content[
                        : self.cfg.retrieval.max_tokens
                    ]

                llm_response = self.qa_chain.invoke(
                    {"query": question, "context": retrieved_docs}
                )

                self.logger.info(f"\nQuestion: {question}")
                self.logger.info(f"\nAnswer: {llm_response['result']}\n")

                data_list.append(
                    {
                        "question": question,
                        "contexts": [
                            " ".join([doc.page_content for doc in retrieved_docs])
                        ],
                        "answer": llm_response["result"],
                    }
                )

                # need to return data_list later for evaluation

                self.answer_file.write(f"{question} - {llm_response['result']}.\n")

            df = pd.DataFrame(data=data_list)
        return Dataset.from_pandas(df=df)

    def run_inference(self):
        self._load_embedding_model()
        self._load_vectordb()
        self._load_prompt_template()
        self._initialize_llm()
        self._create_retriever()
        self._create_qa_chain()
        self._open_questions()
        return self._infer()
