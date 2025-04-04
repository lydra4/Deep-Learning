import logging
from typing import List, Optional, Tuple

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

    def chat_response(
        self, history: List[Tuple[str, str]], question: str
    ) -> Tuple[List[Tuple[str, str]], str]:
        response = self.inference_pipeline.qa_chain.invoke({"query": question})
        answer = response["result"]
        history.append((question, answer))

        return history, ""

    def launch_app(self):
        with gr.Blocks() as demo:
            gr.Markdown(
                "Game of Thrones Chatbot \n\n Ask me anything about Game of Thrones!"
            )
            chatbot = gr.Chatbot()
            user_input = gr.Textbox(
                placeholder="Type your question here.", show_label=False
            )

            def respond(history, question):
                updated_history, _ = self.chat_response(
                    history=history, question=question
                )
                return updated_history

            user_input.submit(respond, [chatbot, user_input], [chatbot]).then(
                lambda: "", [], [user_input]
            )

        demo.launch(server_name="0.0.0.0", server_port=7860)
