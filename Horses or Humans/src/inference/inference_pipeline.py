import logging
import os
from typing import Optional, Tuple

import cv2
import numpy as np
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from PIL import Image
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
        self.model_name: Optional[str] = None
        self.model: Optional[nn.Module] = None

    def initialize_model_with_weights(self) -> None:
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
            number_features = self.model.classifier[1].in_features
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
        self.logger.info("Checkpoint loaded successfully.")
        self.model.to(self.device)
        self.model.eval()

    def generate_cam(
        self, image_tensor: torch.Tensor, original_image: np.ndarray, target_class: int
    ) -> np.ndarray:
        feature_maps = None
        gradients = None

        def forward_hook(module, input, output):
            nonlocal feature_maps
            feature_maps = output.detach()

        def backward_hook(module, grad_in, grad_out):
            nonlocal gradients
            gradients = grad_out[0].detach()

        handle_fwd = self.model.features[-1].register_forward_hook(forward_hook)
        handle_bwd = self.model.features[-1].register_backward_hook(backward_hook)

        output = self.model(image_tensor)
        self.model.zero_grad()
        class_score = output[0, target_class]
        class_score.backward()

        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

        for i in range(feature_maps.shape[1]):
            feature_maps[0, i, :, :] *= pooled_gradients[i]

        cam = torch.sum(feature_maps, dim=1).squeeze()
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / cam.max()
        cam = cam.cpu().numpy()
        cam = cv2.resize(cam, (original_image.shape[1], original_image.shape[0]))
        cam = np.uint8(255 * cam)
        cam = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
        superimposed_img = cv2.addWeighted(original_image, 0.5, cam, 0.5, 0)

        handle_fwd.remove()
        handle_bwd.remove()

        return superimposed_img

    def classify_and_generate_cam(self, image: Image.Image) -> Tuple[str, np.ndarray]:
        image_tensor = self.transform(img=image).unsqueeze(0).to(self.device)
        outputs = self.model(image_tensor)
        prediction = torch.argmax(outputs, dim=1).item()
        label = "horse" if prediction == 0 else "human"

        cam_image = self.generate_cam(
            image_tensor=image_tensor,
            original_image=np.array(image),
            target_class=prediction,
        )

        return label, cam_image

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

    def run_inference(self):
        """Executes the full inference pipeline.

        This method:
        - Loads and splits the dataset.
        - Initializes the model and loads weights.
        - Runs inference on the test set and logs performance.
        """
        self._set_transforms()
        self._load_dataset_from_folder()
        self.initialize_model_with_weights()
        self._run_infer(data_loader=self.test_loader)

    def batch_infer(self):
        self._set_transforms()
        self.initialize_model_with_weights()
