import logging
from typing import List, Optional, Tuple

import gradio as gr
import omegaconf
from inference.inference_pipeline import InferencePipeline
from langfuse.decorators import observe


class GradioApp:
    def __init__(
        self, cfg: omegaconf.DictConfig, logger: Optional[logging.Logger] = None
    ) -> None:
        self.cfg = cfg
        self.logger = logger or logging.getLogger(__name__)
        self.first_part: Optional[str] = None
        self.second_part: Optional[str] = None
        self.inference_pipeline = InferencePipeline(cfg=cfg, logger=self.logger)

        self.inference_pipeline.load_embedding_model()
        self.inference_pipeline._load_vectordb()
        self.inference_pipeline._initialize_llm()
        self.inference_pipeline._create_retriever()
        self.inference_pipeline._create_qa_chain()

    def _load_captions(self):
        with open(file=self.cfg.gradio.captions_path, mode="r", encoding="utf-8") as f:
            content = f.read()

        parts = content.strip().split("---")

        self.first_part = parts[0]
        self.second_part = parts[1]

    @observe()
    def chat_response(
        self, history: List[Tuple[str, str]], question: str
    ) -> Tuple[List[Tuple[str, str]], str]:
        response = self.inference_pipeline.qa_chain.invoke(
            {"query": question},
            config={"callbacks": [self.inference_pipeline.langfuse_handler]},
        )
        answer = response["result"]
        history.append((question, answer))

        return history, ""

    @observe()
    def launch_app(self):
        self._load_captions()

        with gr.Blocks() as demo:
            gr.Markdown(self.first_part)
            chatbot = gr.Chatbot()
            user_input = gr.Textbox(placeholder=self.second_part, show_label=False)

            def respond(history, question):
                updated_history, _ = self.chat_response(
                    history=history, question=question
                )
                return updated_history

            user_input.submit(respond, [chatbot, user_input], [chatbot]).then(
                lambda: "", [], [user_input]
            )

        demo.launch()
