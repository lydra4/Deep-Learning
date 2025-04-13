import logging
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader
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
    """
    A training pipeline for image classification using PyTorch.

    This class handles loading datasets, initializing models, training, evaluation, and logging with MLflow.

    Attributes:
        cfg (dict): Configuration dictionary containing model, dataset, training, and MLflow parameters.
        logger (Optional[logging.Logger]): Logger instance for logging messages.
        device (torch.device): The device (CPU or GPU) to use for model training.
        dataset (Optional[Dataset]): Dataset object for image data.
        train_loader (Optional[DataLoader]): DataLoader for training data.
        test_loader (Optional[DataLoader]): DataLoader for test data.
        model (Optional[nn.Module]): Model to be trained.
        criterion (Optional[nn.Module]): Loss function used in training.
        optimizer (Optional[optim.Optimizer]): Optimizer used for training.
        mlflow_init_status (Optional[bool]): Status of MLflow initialization.
        mlflow_run (Optional[mlflow.ActiveRun]): MLflow run object.
        epoch (Optional[int]): Current epoch in the training process.
    """

    def __init__(
        self, cfg: dict, logger: Optional[logging.Logger], device: torch.device
    ):
        """
        Initializes the TrainingPipeline.

        Args:
            cfg (dict): Configuration dictionary containing model, dataset, training, and MLflow parameters.
            logger (Optional[logging.Logger]): Logger instance for logging messages. If None, a default logger is created.
            device (torch.device): Torch device to use for model training (CPU or GPU).
        """
        self.cfg = cfg
        self.logger = logger or logging.getLogger(__name__)
        self.device = device
        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None
        self.test_loader: Optional[DataLoader] = None
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.mlflow_init_status = None
        self.mlflow_run = None
        self.epoch = None

    def _load_dataset_from_folder(self):
        """
        Loads and applies transformations to the image dataset.

        Loads the dataset using torchvision's ImageFolder from the `path_to_processed_data` and applies
        resizing and tensor conversion.

        Logs dataset loading information and class-to-index mapping.
        """
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

        self.train_loader = DataLoader(
            dataset=datasets.ImageFolder(
                root=os.path.join(self.cfg.environ.path_to_processed_data, "train"),
                transform=transform,
            ),
            batch_size=self.cfg.batch_size,
            shuffle=True,
        )

        self.val_loader = DataLoader(
            dataset=datasets.ImageFolder(
                root=os.path.join(self.cfg.environ.path_to_processed_data, "val"),
                transform=transform,
            ),
            batch_size=self.cfg.batch_size,
            shuffle=False,
        )

        self.test_loader = DataLoader(
            dataset=datasets.ImageFolder(
                root=os.path.join(self.cfg.environ.path_to_processed_data, "test"),
                transform=transform,
            ),
            batch_size=self.cfg.batch_size,
            shuffle=False,
        )

        self.logger.info("\nSuccessfully loaded Train, Val and Test.\n")

    def _instantiate_model(self):
        """
        Initializes the model based on the specified architecture.

        Supports ConvNeXt, EfficientNet-B0, and ResNet-18. Replaces the final classification layer
        with one compatible with the number of output classes defined in the config.

        Raises:
            ValueError: If an unsupported model name is provided in the configuration.
        """
        try:
            self.logger.info(f"\nLoading {self.cfg.model} model.\n")
            if self.cfg.model.lower().strip() == "convnext":
                self.model = models.convnext_small(
                    weights=ConvNeXt_Small_Weights.DEFAULT
                )
                number_features = self.model.classifier[2].in_features
                self.model.classifier[2] = nn.Linear(
                    in_features=number_features, out_features=self.cfg.out_features
                )

            elif self.cfg.model.lower().strip() == "efficientnet":
                self.model = efficientnet_b0(weights=EfficientNet_B0_Weights)
                number_features = self.model.classifier[1].in_features
                self.model.classifier[1] = nn.Linear(
                    in_features=number_features, out_features=self.cfg.out_features
                )

            elif self.cfg.model.lower().strip() == "resnet-18":
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
        """
        Sets the loss function and optimizer.

        Uses CrossEntropyLoss and the Adam optimizer with a learning rate from the config.
        """
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            params=self.model.parameters(), lr=self.cfg["learning_rate"]
        )

    def _setup_mlflow(self):
        """
        Initializes MLflow for experiment tracking.

        Calls the `mlflow_init` utility function to setup the MLflow run and logs the status.
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
            self.logger.info("\nMLflow initialized.\n")
        else:
            self.logger.error("MLflow initialization failed.")

    def _evaluate(self, data_loader):
        self.model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return 100 * correct / total

    def _save_checkpoint(self):
        """
        Saves model checkpoints to disk.

        Saves the model’s state_dict to the directory defined in `checkpoint_save_path`
        under a subfolder named after the model.
        """
        checkpoint_path = os.path.join(
            self.cfg.checkpoint_save_path,
            self.cfg.model,
            f"model_epoch_{self.epoch + 1}.pth",
        )
        torch.save(self.model.state_dict(), checkpoint_path)
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")

    def _train_model(self):
        """
        Trains the model over a number of epochs.

        For each epoch, computes training loss and accuracy, logs metrics to MLflow,
        and saves checkpoints periodically.
        """
        os.makedirs(
            name=os.path.join(self.cfg.checkpoint_save_path, self.cfg.model),
            exist_ok=True,
        )
        self.logger.info(f"\nTraining for {self.cfg['epochs']} epochs.\n")
        self.model.train()

        for self.epoch in range(self.cfg["epochs"]):
            self.logger.info(f"Starting Epoch {self.epoch + 1}/{self.cfg['epochs']}.\n")

            self.model.train()
            running_loss, train_correct, train_total = 0, 0, 0

            for images, labels in tqdm(
                iterable=self.train_loader,
                desc="Training",
                unit="Batch",
            ):
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

            train_accuracy = 100 * train_correct / train_total
            val_accuracy = self._evaluate(self.val_loader)
            self.logger.info(
                f"\nFor Epoch {self.epoch + 1}, Train Accuracy:{train_accuracy:.2f}%, Val Accuracy:{val_accuracy:.2f}%.\n"
            )

            if self.mlflow_init_status:
                mlflow_log(
                    mlflow_init_status=self.mlflow_init_status,
                    log_function="log_metric",
                    key="Train Accuracy",
                    value=train_accuracy,
                    step=self.epoch,
                )
                mlflow_log(
                    mlflow_init_status=self.mlflow_init_status,
                    log_function="log_metric",
                    key="Val Accuracy",
                    value=val_accuracy,
                    step=self.epoch,
                )

            if (self.epoch + 1) % 5 == 0:
                self._save_checkpoint()

    def run_training_pipeline(self):
        """
        Runs the entire training pipeline, including dataset loading, model training, and MLflow logging.

        This method orchestrates the following steps:
        1. Loading the dataset
        2. Splitting the dataset
        3. Instantiating the model
        4. Setting the loss function and optimizer
        5. Setting up MLflow
        6. Training the model
        """
        self._load_dataset_from_folder()
        self._instantiate_model()
        self._set_criterion_optimizer()
        self._setup_mlflow()
        self._train_model()
