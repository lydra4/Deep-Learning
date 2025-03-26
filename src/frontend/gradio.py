import logging
from typing import Optional

import gradio as gr
import omegaconf

from inference.inference_pipeline import InferencePipeline


class GradioApp:
    def __init__(
        self, cfg: omegaconf.DictConfig, logger: Optional[logging.Logger] = None
    ) -> None:
        self.cfg = cfg
        self.logger = logger or logging.getLogger(__name__)
        self.inference_pipeline = InferencePipeline(cfg=cfg, logger=self.logger)

        self.inference_pipeline.load_embedding_model()
        self.inference_pipeline._load_vectordb()
        self.inference_pipeline._initialize_llm()
        self.inference_pipeline._create_retriever()
        self.inference_pipeline._create_qa_chain()

    def chat_response(self, question: str) -> str:
        response = self.inference_pipeline.qa_chain.invoke({"query": question})
        return response["result"]

    def launch_app(self):
        iface = gr.Interface(
            fn=self.chat_response,
            inputs="text",
            outputs="text",
            title="Game of Thrones Chatbot",
            description="Ask me anything about Game of Thrones",
            theme="dark",
        )
        iface.launch()
