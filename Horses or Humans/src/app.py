import logging
import os

import gradio as gr
import hydra
import omegaconf
import torch
from inference.inference_pipeline import InferencePipeline
from PIL import Image
from utils.general_utils import setup_logging
from utils.seed_utils import fix_seed


@hydra.main(version_base=None, config_path="../conf", config_name="inference.yaml")
def main(cfg: omegaconf.DictConfig):
    logger = logging.getLogger(__name__)
    logging.info("Setting up logging configuration.\n")
    setup_logging(
        logging_config_path=os.path.join(
            hydra.utils.get_original_cwd(), "conf", "logging.yaml"
        )
    )

    fix_seed(seed=cfg.environ.seed)

    if cfg.environ.device < 0:
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{cfg.environ.device}")

    inference_pipeline = InferencePipeline(cfg=cfg, logger=logger, device=device)
    inference_pipeline.batch_infer()

    def gradio_inferface(image: Image.Image):
        return inference_pipeline.classify_and_generate_cam(image=image)

    gr.Interface(
        fn=gradio_inferface,
        inputs=gr.Image(type="pil", label="Upload an image of a horse or human"),
        outputs=[
            gr.Label(num_top_classes=1, label="Prediction"),
            gr.Image(type="numpy", label="Class Activation Map"),
        ],
        title="Horse vs Human Classifer with CAM",
        description="Upload an image to classify it as either a horse or human",
        theme="dark",
    ).launch()


if __name__ == "__main__":
    main()
