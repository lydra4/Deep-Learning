import logging
import os

import omegaconf
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import (
    ConvNeXt_Small_Weights,
    EfficientNet_B0_Weights,
    ResNet18_Weights,
    efficientnet_b0,
    resnet18,
)
from tqdm import tqdm
from training_pipelines.training_pipelines import TrainingPipeline


class InferencePipeline(TrainingPipeline):
    """Pipeline for performing inference using a trained model.

    This class extends the `TrainingPipeline` to handle loading a saved
    model checkpoint and evaluating it on a test dataset.

    Attributes:
        model_name (str): Name of the model architecture.
        model (torch.nn.Module): Initialized model for inference.
    """

    def __init__(
        self, cfg: omegaconf.DictConfig, logger: logging.Logger, device: torch.device
    ) -> None:
        """Initializes the InferencePipeline.

        Args:
            cfg (omegaconf.DictConfig): Configuration object with dataset and environment settings.
            logger (logging.Logger): Logger instance for recording logs.
            device (torch.device): Torch device (CPU or GPU) used for running inference.
        """
        super().__init__(cfg=cfg, logger=logger, device=device)

    def _initialize_model_with_weights(self) -> None:
        """Initializes the model with pretrained weights and loads the saved checkpoint.

        This method:
        - Loads the appropriate model architecture based on the checkpoint path.
        - Modifies the classifier head to match the number of output features.
        - Loads the saved state dictionary from the checkpoint.
        - Moves the model to the specified device and sets it to evaluation mode.

        Raises:
            ValueError: If the model architecture is not supported.
        """
        self.logger.info(f"Loading model checkpoint from {self.cfg.checkpoint_path}.\n")
        self.model_name = os.path.basename(os.path.dirname(self.cfg.checkpoint_path))
        out_features = self.cfg.out_features

        self.logger.info(f"Initializing {self.model_name}.\n")
        if self.model_name.lower() == "convnet":
            self.model = models.convnext_small(weights=ConvNeXt_Small_Weights.DEFAULT)
            number_features = self.model.classifier[2].in_features
            self.model.classifier[2] = nn.Linear(
                in_features=number_features, out_features=out_features
            )

        elif self.model_name.lower() == "efficientnet":
            self.model = efficientnet_b0(weights=EfficientNet_B0_Weights)
            number_features = self.model.classified[1].in_features
            self.model.classifier[1] = nn.Linear(
                in_features=number_features, out_features=out_features
            )

        elif self.model_name.lower() == "resnet-18":
            self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
            number_features = self.model.fc.in_features
            self.model.fc = nn.Linear(
                in_features=number_features, out_features=out_features
            )

        else:
            raise ValueError(f"Unsupported Model: {self.model_name}.")

        checkpoint = torch.load(self.cfg.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.model.eval()

    def _run_infer(self, data_loader: torch.utils.data.DataLoader) -> None:
        """Performs inference on the test dataset and logs the accuracy.

        Args:
            data_loader (torch.utils.data.DataLoader): DataLoader object for the test dataset.
        """
        correct, total = 0, 0

        with torch.no_grad():
            for images, labels in tqdm(data_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                predicted_class = torch.argmax(input=outputs, dim=1)

                correct += (predicted_class == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total
        self.logger.info(f"Accuracy for {self.model_name}: {accuracy}.\n")

    def perform_inference(self):
        """Executes the full inference pipeline.

        This method:
        - Loads and splits the dataset.
        - Initializes the model and loads weights.
        - Runs inference on the test set and logs performance.
        """
        self._load_dataset_from_folder()
        self._initialize_model_with_weights()
        self._run_infer(data_loader=self.test_loader)
