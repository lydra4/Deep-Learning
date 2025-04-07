import json
import logging
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.models import (
    ConvNeXt_Small_Weights,
    EfficientNet_B0_Weights,
    ResNet18_Weights,
    efficientnet_b0,
    resnet18,
)
from tqdm import tqdm
from utils.general_utils import mlflow_init, mlflow_log


class TrainingPipeline:
    """Handles the entire training pipeline including dataset loading, model training, and MLflow logging."""

    def __init__(
        self, cfg: dict, logger: Optional[logging.Logger], device: torch.device
    ):
        """
        Initializes the training pipeline with the provided configuration and logger.

        Args:
            cfg (dict): Configuration dictionary containing necessary parameters for training.
            logger (Optional[logging.Logger]): Logger object for logging training information. If None, a default logger is used.
        """
        self.cfg = cfg
        self.logger = logger or logging.getLogger(__name__)
        self.device = device
        self.dataset = None
        self.train_loader = None
        self.test_loader = None
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.mlflow_init_status = None
        self.mlflow_run = None
        self.epoch = None

    def _load_dataset(self):
        """Loads and transforms the dataset from the specified directory."""
        transform = transforms.Compose(
            [
                transforms.Resize(
                    (self.cfg["transform_resize"], self.cfg["transform_resize"])
                ),
                transforms.ToTensor(),
            ]
        )

        self.logger.info(
            f"Loading dataset from {self.cfg.environ.path_to_processed_data}.\n"
        )
        self.dataset = datasets.ImageFolder(
            root=self.cfg.environ.path_to_processed_data, transform=transform
        )
        self.logger.info("Successfully loaded dataset.")
        self.logger.info(
            f"Class to Index Mapping: {json.dumps(self.dataset.class_to_idx, indent=4)}"
        )

    def _split_dataset(self):
        """
        Splits the dataset into training and testing sets based on the train ratio.

        Returns:
            tuple: A tuple containing the train DataLoader and test DataLoader.
        """
        train_size = int(self.cfg["train_ratio"] * len(self.dataset))
        test_size = len(self.dataset) - train_size
        self.logger.info(
            f"Train size:{self.cfg['train_ratio']}\n Test size:{round((1 - self.cfg['train_ratio']), 2)}.\n"
        )

        train_dataset, test_dataset = random_split(
            dataset=self.dataset,
            lengths=[train_size, test_size],
            generator=torch.Generator().manual_seed(self.cfg.environ.seed),
        )

        self.train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.cfg["batch_size"],
            shuffle=True,
        )
        self.test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=self.cfg["batch_size"],
            shuffle=False,
        )

    def _instantiate_model(self):
        """Instantiates the model, replacing the final layer with a custom output layer."""
        try:
            self.logger.info(f"Loading {self.cfg.model} model.\n")
            if self.cfg.model.lower() == "convnext":
                self.model = models.convnext_small(
                    weights=ConvNeXt_Small_Weights.DEFAULT
                )
                number_features = self.model.classifier[2].in_features
                self.model.classifier[2] = nn.Linear(
                    in_features=number_features, out_features=self.cfg.out_features
                )

            elif self.cfg.model.lower() == "efficientnet":
                self.model = efficientnet_b0(weights=EfficientNet_B0_Weights)
                number_features = self.model.classifier[1].in_features
                self.model.classifier[1] = nn.Linear(
                    in_features=number_features, out_features=self.cfg.out_features
                )

            elif self.cfg.model.lower() == "resnet-18":
                self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
                number_features = self.model.fc.in_features
                self.model.fc = nn.Linear(
                    in_features=number_features, out_features=self.cfg.out_features
                )

            else:
                raise ValueError(f"Unsupported model: {self.cfg.model}.")

        except Exception as e:
            self.logger.error(f"{self.cfg.model} model failed to load: {e}.")

        self.model.to(self.device)
        self.logger.info(f"Model loaded to {str(self.device).upper()}.\n")

    def _set_criterion_optimizer(self):
        """Sets the loss function (CrossEntropyLoss) and optimizer (Adam)."""
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            params=self.model.parameters(), lr=self.cfg["learning_rate"]
        )

    def _setup_mlflow(self):
        """
        Initializes MLflow with the configuration provided in the setup.

        Logs whether the MLflow initialization was successful.
        """
        mlflow_args = {
            "mlflow_tracking_uri": self.cfg.environ.mlflow.mlflow_tracking_uri,
            "mlflow_exp_name": self.cfg.environ.mlflow.mlflow_exp_name,
            "mlflow_run_name": self.cfg.model,
        }
        self.mlflow_init_status, self.mlflow_run = mlflow_init(
            args=mlflow_args,
            run_name=self.cfg.model,
            setup_mlflow=self.cfg.environ.mlflow.setup_mlflow,
            autolog=self.cfg.environ.mlflow.autolog,
        )
        if self.mlflow_init_status:
            self.logger.info("MLflow initialized")
        else:
            self.logger.error("MLflow initialization failed.")

    def _train_model(self):
        """
        Trains the model for the specified number of epochs, logging metrics and saving checkpoints periodically.
        """
        os.makedirs(name=self.cfg["checkpoint_save_path"], exist_ok=True)
        self.logger.info(f"Training for {self.cfg['epochs']} epochs.\n")
        self.model.train()

        for self.epoch in range(self.cfg["epochs"]):
            self.logger.info(f"Starting Epoch {self.epoch + 1}/{self.cfg['epochs']}.\n")

            running_loss = 0
            correct = 0
            total = 0

            for images, labels in tqdm(
                iterable=self.train_loader,
                desc=f"Epoch {self.epoch + 1}/{self.cfg['epochs']}",
                unit="Images",
            ):
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total

            if self.mlflow_init_status:
                mlflow_log(
                    self.mlflow_init_status,
                    "log_metric",
                    key="Accuracy",
                    value=accuracy,
                    step=self.epoch,
                )

            if (self.epoch + 1) % 5 == 0:
                self._save_checkpoint()

            self.logger.info(
                f"Epoch {self.epoch + 1}/{self.cfg['epochs']}, Accuracy: {accuracy:.2f}%.\n"
            )

    def _save_checkpoint(self):
        """
        Saves the model checkpoint after each epoch.

        The checkpoint is saved in the directory specified by `checkpoint_save_path`.
        """
        os.makedirs(
            name=os.path.join(self.cfg.checkpoint_save_path, self.cfg.model),
            exist_ok=True,
        )
        checkpoint_path = os.path.join(
            self.cfg.checkpoint_save_path,
            self.cfg.model,
            f"model_epoch_{self.epoch + 1}.pth",
        )
        torch.save(self.model.state_dict(), checkpoint_path)
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")

    def run_training_pipeline(self):
        """
        Runs the entire training pipeline, including dataset loading, model training, and MLflow logging.

        This method orchestrates the following:
        1. Loading the dataset
        2. Splitting the dataset
        3. Instantiating the model
        4. Setting the loss function and optimizer
        5. Setting up MLflow
        6. Training the model
        """
        self._load_dataset()
        self._split_dataset()
        self._instantiate_model()
        self._set_criterion_optimizer()
        self._setup_mlflow()
        self._train_model()
