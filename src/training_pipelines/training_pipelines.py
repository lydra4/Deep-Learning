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
from torchvision.models import ConvNeXt_Small_Weights
from tqdm import tqdm
from utils.general_utils import mlflow_init, mlflow_log


class TrainingPipeline:
    """Handles the entire training pipeline including dataset loading, model training, and MLflow logging."""

    def __init__(self, cfg: dict, logger: Optional[logging.Logger]):
        """
        Initializes the training pipeline with the provided configuration and logger.

        Args:
            cfg (dict): Configuration dictionary containing necessary parameters for training.
            logger (Optional[logging.Logger]): Logger object for logging training information. If None, a default logger is used.
        """
        self.cfg = cfg
        self.logger = logger or logging.getLogger(__name__)
        self.dataset = None
        self.train_loader = None
        self.test_loader = None
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.mlflow_init_status = None
        self.mlflow_run = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

        self.logger.info(f"Loading dataset from {self.cfg['path_to_processed_data']}")
        self.dataset = datasets.ImageFolder(
            root=self.cfg["path_to_processed_data"], transform=transform
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
            generator=torch.Generator().manual_seed(self.cfg["seed"]),
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
        if self.cfg.model == "convnext":
            try:
                self.logger.info(f"Loading {self.cfg.model} model.\n")
                self.model = models.convnext_small(
                    weights=ConvNeXt_Small_Weights.DEFAULT
                )
                number_features = self.model.classifier[2].in_features
                self.model.classifier[2] = nn.Linear(
                    in_features=number_features, out_features=self.cfg["out_features"]
                )
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
            "mlflow_tracking_uri": self.cfg["mlflow_tracking_uri"],
            "mlflow_exp_name": self.cfg["mlflow_exp_name"],
            "mlflow_run_name": self.cfg["mlflow_run_name"],
        }
        self.mlflow_init_status, self.mlflow_run = mlflow_init(
            args=mlflow_args,
            run_name=self.cfg["mlflow_run_name"],
            setup_mlflow=self.cfg["setup_mlflow"],
            autolog=self.cfg["autolog"],
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
        checkpoint_path = os.path.join(
            self.cfg["checkpoint_save_path"], f"model_epoch_{self.epoch + 1}.pth"
        )
        torch.save(self.model.state_dict(), checkpoint_path)
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")

    def _load_checkpoint(self):
        """
        Loads the model checkpoint from the specified path in the configuration.

        This method checks if the checkpoint file exists at the given path. If the file is found,
        it loads the model state dictionary from the checkpoint and moves the model to the specified device.

        If the checkpoint is not found, an error is logged.

        Attributes:
            cfg (dict): Configuration dictionary containing the checkpoint load path.
            logger (logging.Logger): Logger instance to log information and errors.
            model (torch.nn.Module): The PyTorch model to load the state dictionary into.
            device (torch.device): The device to which the model is moved (e.g., CPU or GPU).
        """
        if not os.path.exists(self.cfg["checkpoint_load_path"]):
            self.logger.error(
                f"Checkpoint not found at {self.cfg['checkpoint_load_path']}"
            )
            return

        self.model.load_state_dict(
            torch.load(self.cfg["checkpoint_load_path"], map_location=self.device)
        )
        self.model.to(self.device)
        self.logger.info(f"Loaded checkpoint: {self.cfg['checkpoint_load_path']}")

    def evaluate_model(self):
        """
        Evaluates the model on the test dataset and logs the accuracy.

        This method loads the test dataset, splits it, instantiates the model, and loads the checkpoint.
        It then switches the model to evaluation mode, performs inference on the test data,
        and calculates the accuracy by comparing the predicted and actual labels.

        Logs the accuracy of the model after evaluation.

        Attributes:
            test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
            model (torch.nn.Module): The PyTorch model to evaluate.
            device (torch.device): The device to perform the evaluation on (e.g., CPU or GPU).
            logger (logging.Logger): Logger instance to log evaluation results.
        """
        self._load_dataset()
        self._split_dataset()
        self._instantiate_model()
        self._load_checkpoint()

        self.model.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in tqdm(
                self.test_loader, desc="Evaluating", unit="Images"
            ):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        self.logger.info(f"Test Accuracy: {accuracy:.2f}%")

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
