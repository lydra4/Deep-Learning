import locale
import logging
import os

import hydra
import torch
from dotenv import load_dotenv
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.prompts.prompt import PromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from utils.general_utils import setup_logging


@hydra.main(version_base=None, config_path="../conf", config_name="inference.yaml")
def main(cfg):
    logger = logging.getLogger(__name__)
    logger.info("Setting up logging configuration")
    setup_logging(
        logging_config_path=os.path.join(
            hydra.utils.get_original_cwd(), "conf", "logging.yaml"
        )
    )

    if not os.path.exists(cfg["embeddings_path"]):
        raise FileNotFoundError(f"The path, {cfg['embeddings_path']}, does not exits.")

    logger.info("Loading Vector DB")

    index_name = os.path.basename(cfg["embeddings_path"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Embedding Model will be loaded to {device.upper()}.")

    model_config = {
        "model_name": cfg["embeddings_model_name"],
        "show_progress": cfg["show_progress"],
        "model_kwargs": {"device": device},
    }

    embedding_model = HuggingFaceInstructEmbeddings(**model_config)

    logger.info(f"Embedding Model loaded to {device.upper()}.")

    vectordb = FAISS.load_local(
        folder_path=cfg["embeddings_path"],
        embeddings=embedding_model,
        index_name=index_name,
        allow_dangerous_deserialization=True,
    )

    logger.info("Successfully loaded")

    with open(
        file=cfg["path_to_template"], mode="r", encoding=locale.getencoding()
    ) as f:
        template = f.read()

    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"],
    )

    load_dotenv()

    llm = ChatOpenAI(
        model=cfg["model"],
        temperature=cfg["temperature"],
        api_key=os.getenv("api_key"),
    )

    retriever = vectordb.as_retriever(
        search_kwargs={"k": cfg["k"], "search_type": cfg["search_type"]}
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type=cfg["chain_type"],
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=cfg["return_source_documents"],
        verbose=cfg["verbose"],
    )

    max_input_tokens = cfg["max_tokens"]

    with open(file=cfg["path_to_qns"], mode="r", encoding=locale.getencoding()) as f:
        qns_list = f.readlines()
        qns_list = [line.rstrip("\n").strip() for line in qns_list]
        f.close()

    logger.info(f"Total number of questions: {len(qns_list)}.\n")

    folder_to_ans = os.path.dirname(cfg["path_to_ans"])
    os.makedirs(name=folder_to_ans, exist_ok=True)

    with open(
        file=cfg["path_to_ans"], mode="w", encoding=locale.getencoding()
    ) as ans_file:
        logger.info(f"Answers will be saved in {cfg['path_to_ans']}.\n")

        data_list = []

        for question in qns_list:
            retrieved_docs = retriever.invoke(question)

            for document in retrieved_docs:
                document.page_content = document.page_content[:max_input_tokens]

            llm_response = qa_chain.invoke(
                {"query": question, "context": retrieved_docs}
            )

            logger.info(f"\nQuestion: {question}")
            logger.info(f"\nAnswer: {llm_response['result']}\n")
            logger.info(
                f"\nContext: {' '.join([doc.page_content for doc in retrieved_docs])}\n"
            )

            data_dict = {
                "question": question,
                "contexts": [" ".join([doc.page_content for doc in retrieved_docs])],
                "answer": llm_response["result"],
            }

            data_list.append(data_dict)

            ans_file.write(f"{question} - {llm_response['result']}.\n")


if __name__ == "__main__":
    main()
