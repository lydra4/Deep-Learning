import locale
import logging
import os
from typing import Optional

import omegaconf
import torch
from dotenv import load_dotenv
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.prompts.prompt import PromptTemplate
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_openai.chat_models import ChatOpenAI
from ragas import EvaluationDataset


class InferencePipeline:
    def __init__(
        self, cfg: omegaconf.DictConfig, logger: Optional[logging.Logger] = None
    ) -> None:
        """
        Initializes the inference pipeline.

        Args:
            cfg (omegaconf.DictConfig): Configuration dictionary for the pipeline.
            logger (Optional[logging.Logger], optional): Logger instance. Defaults to None.
        """
        self.cfg = cfg
        self.logger = logger or logging.getLogger(__name__)
        self.embedding_model: Optional[HuggingFaceInstructEmbeddings] = None
        self.vectordb: Optional[FAISS] = None
        self.prompt: Optional[PromptTemplate] = None
        self.llm: Optional[ChatOpenAI] = None
        self.retriever = None
        self.qa_chain = None
        self.qns_list: Optional[list] = None
        self.ans_list: Optional[list] = None
        self.answer_file: Optional[str] = None

    def load_embedding_model(self):
        """
        Loads the embedding model specified in the configuration.

        Raises:
            Exception: If the embedding model fails to load.
        """
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

        return self.embedding_model

    def _load_vectordb(self):
        """
        Loads the FAISS vector database from the specified path.

        Raises:
            FileNotFoundError: If the vector database path does not exist.
            Exception: If loading the vector database fails.
        """
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
        """
        Loads the prompt template from the specified file path.

        Raises:
            Exception: If the prompt template file cannot be loaded.
        """
        self.logger.info("Loading Prompt Template")

        try:
            with open(
                file=self.cfg.path_to_template,
                mode="r",
                encoding="utf-8",
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
        """
        Initializes the language model (LLM) with the specified API key.

        Raises:
            ValueError: If the API key is missing.
            Exception: If initializing the LLM fails.
        """
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
        """
        Creates a retriever for document retrieval based on the vector database.
        """
        self.retriever = self.vectordb.as_retriever(
            search_kwargs={
                "k": self.cfg.retrieve.k,
                "search_type": self.cfg.retrieve.search_type,
            }
        )

    def _create_qa_chain(self):
        """
        Creates a RetrievalQA chain using the initialized LLM and retriever.
        """
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type=self.cfg.retrieval.chain_type,
            retriever=self.retriever,
            chain_type_kwargs={"prompt": self.prompt},
            return_source_documents=self.cfg.retrieval.return_source_documents,
            verbose=self.cfg.retrieval.verbose,
        )

    def _open_questions(self):
        """
        Loads the list of questions from the specified file.

        Raises:
            Exception: If the questions file cannot be loaded.
        """
        self.logger.info("Loading questions...")

        try:
            with open(
                file=self.cfg.llm.path_to_qns, mode="r", encoding=locale.getencoding()
            ) as f:
                lines = [line.rstrip("\n").strip() for line in f.readlines()]

                self.qns_list = [line.split(" - ", 1)[0].strip() for line in lines]
                self.ground_truth = [line.split(" - ", 1)[1].strip() for line in lines]

            self.logger.info(f"Loaded {len(self.qns_list)} questions.")

        except Exception as e:
            self.logger.error(f"Failed to load questions: {e}")
            raise

    def _infer(self):
        """
        Runs inference on the loaded questions, retrieves relevant contexts,
        and generates answers using the LLM.

        Returns:
            Dataset: A Hugging Face dataset containing the questions, contexts, and answers.

        Raises:
            Exception: If inference fails.
        """
        folder_to_answers = os.path.dirname(self.cfg.llm.path_to_ans)
        os.makedirs(name=folder_to_answers, exist_ok=True)

        self.logger.info(f"Saving answers to {self.cfg.llm.path_to_ans}")
        data_list = []

        with open(
            file=self.cfg.llm.path_to_ans,
            mode="w",
            encoding="utf-8",
            newline="\n",
        ) as self.answer_file:
            for question, ground_truth in zip(self.qns_list, self.ground_truth):
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
                        "user_input": question,
                        "reference": ground_truth,
                        "response": llm_response["result"],
                        "retrieved_contexts": [
                            " ".join([doc.page_content for doc in retrieved_docs])
                        ],
                    }
                )

                self.answer_file.write(f"{question} - {llm_response['result']}.\n")

        return EvaluationDataset.from_list(data=data_list), len(self.qns_list)

    def run_inference(self):
        """
        Executes the full inference pipeline.

        Returns:
            Dataset: A dataset containing the questions, retrieved contexts, and generated answers.
        """
        self.load_embedding_model()
        self._load_vectordb()
        self._load_prompt_template()
        self._initialize_llm()
        self._create_retriever()
        self._create_qa_chain()
        self._open_questions()
        return self._infer()
