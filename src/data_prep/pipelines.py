import logging
import os
import random
from collections import Counter
from typing import Optional

import omegaconf
from PIL import Image
from torchvision import datasets, transforms
from tqdm import tqdm


class ImagePipeline:
    """Pipeline for processing image datasets, including augmentation and copying.

    This class processes image datasets by:
    1. Calculating class distributions.
    2. Creating necessary directories.
    3. Copying original images to a new location while applying augmentation.
    4. Performing additional data augmentation on minority classes to balance the dataset.

    Attributes:
        cfg (dict): Configuration dictionary.
        logger (logging.Logger): Logger for tracking progress.
        dataset (datasets.ImageFolder): Dataset loaded using torchvision.
        class_counts (Counter): Dictionary storing the count of images per class.
        max_value (int): Maximum number of images in any class.
        processed_path (str): Path to store processed images.
    """

    def __init__(
        self, cfg: omegaconf.DictConfig, logger: Optional[logging.Logger]
    ) -> None:
        """Initializes the ImagePipeline with configuration and logging.

        Args:
            cfg (dict): Configuration dictionary containing paths and augmentation settings.
            logger (Optional[logging.Logger]): Logger for tracking progress. Defaults to None.
        """
        self.cfg = cfg
        self.logger = logger or logging.getLogger(__name__)
        self.dataset = datasets.ImageFolder(root=self.cfg.path_to_unprocessed_data)
        self.class_counts = Counter(
            [self.dataset.classes[label] for _, label in self.dataset.samples]
        )
        self.max_value: Optional[int] = max(self.class_counts.values())
        self.processed_path: Optional[str] = self.cfg.path_to_processed_data

    def _get_class_distribution(self) -> None:
        """Calculates the number of images per class in the dataset."""
        self.logger.info(f"Class distribution: {self.class_counts}.\n")

    def _data_augmentation(self, image: Image.Image) -> Image.Image:
        """Performs data augmentation on minority classes to balance the dataset."""
        augment_transforms = transforms.Compose(
            [
                transforms.ColorJitter(
                    brightness=self.cfg.color_jitter_brightness,
                    contrast=self.cfg.color_jitter_contrast,
                    saturation=self.cfg.color_jitter_saturation,
                ),
                transforms.GaussianBlur(
                    kernel_size=(self.cfg.kernel_size, self.cfg.kernel_size),
                    sigma=(self.cfg.lower_sigma, self.cfg.upper_sigma),
                ),
                transforms.RandomResizedCrop(
                    size=(self.cfg.size, self.cfg.size),
                    scale=(self.cfg.lower_scale, self.cfg.upper_scale),
                ),
                transforms.ToTensor(),
                transforms.ToPILImage(),
            ]
        )

        return augment_transforms(img=image)

    def _augment_and_save(self) -> None:
        class_image_count = Counter()

        self.logger.info(f"Copying images to {self.cfg.path_to_processed_data}\n")
        for img_path, label in tqdm(self.dataset.samples, desc="Copying Images"):
            class_name = self.dataset.classes[label]
            save_dir = os.path.join(self.cfg.path_to_processed_data, class_name)
            os.makedirs(name=save_dir, exist_ok=True)

            image = Image.open(img_path).convert("RGB")
            augmented_image = self._data_augmentation(image=image)

            save_path = os.path.join(save_dir, os.path.basename(img_path))
            augmented_image.save(save_path)

            class_image_count[class_name] += 1

        self.logger.info(f"Copying completed, count: {dict(class_image_count)}\n")

    def _balance_minority_class(self) -> None:
        for class_name, count in self.class_counts.items():
            if count < self.max_value:
                self.logger.info(f"Augmenting {class_name} class.\n")

                class_dir = os.path.join(self.cfg.path_to_unprocessed_data, class_name)
                save_dir = os.path.join(self.processed_path, class_name)
                os.makedirs(name=save_dir, exist_ok=True)

                image_paths = [
                    os.path.join(class_dir, img) for img in os.listdir(class_dir)
                ]
                num_images_needed = self.max_value - count

                selected_paths = random.choices(image_paths, k=num_images_needed)

                for i, img_path in enumerate(
                    tqdm(selected_paths, desc="Augmenting Images"), start=1
                ):
                    original_image = Image.open(img_path).convert("RGB")
                    augmented_image = self._data_augmentation(image=original_image)

                    filename_wo_extension, extension = os.path.splitext(
                        os.path.basename(img_path)
                    )

                    save_path = os.path.join(
                        save_dir, f"{filename_wo_extension}_aug_{i}.{extension}"
                    )
                    augmented_image.save(save_path)

                self.logger.info(
                    f"Augmented {class_name} by {num_images_needed} images.\n"
                )

    def run_pipeline(self) -> None:
        """Runs the entire image processing pipeline: distribution, directory setup, copying, and augmentation."""
        self._get_class_distribution()
        self._augment_and_save()
        self._balance_minority_class()
