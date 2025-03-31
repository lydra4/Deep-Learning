import logging
import os
import random
from collections import Counter
from typing import Optional
import time
from multiprocessing import Pool, cpu_count

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
    
    def _process_image(self, args):
        image_path, label = args
        class_name = self.dataset.classes[label]
        save_dir = os.path.join(self.cfg.path_to_processed_data, class_name)
        os.makedirs(name=save_dir, exist_ok=True)

        try:
            image = Image.open(image_path).convert("RGB")
            augmented_image = self._data_augmentation(image=image)
            save_path = os.path.join(save_dir, os.path.basename(image_path))
            augmented_image.save(save_path)
            return class_name
        
        except Exception as e:
            self.logger.error(f"Error processing {image_path}: {e}.")
            return None

    def _augment_and_save(self) -> None:
        self.logger.info(f"Copying images to {self.cfg.path_to_processed_data}\n")

        if self.cfg.use_multiprocessing:
            with Pool(processes=cpu_count()) as pool:
                results = list(tqdm(pool.imap(self._process_image, self.dataset.samples), total=len(self.dataset.samples), desc="Copying Images"))

        else:
            results = [self._process_image(sample) for sample in tqdm(self.dataset.samples, desc="Copying Images")]

        class_image_count = Counter(filter(None, results))
        self.logger.info(f"Copying completed, count: {dict(class_image_count)}\n")

    def _process_augmentation(self, args):
        image_path, save_dir, class_name, i = args

        try:
            original_image = Image.open(image_path).convert("RGB")
            augmented_image = self._data_augmentation(image=original_image)

            filename_wo_extension, extension = os.path.splitext(os.path.basename(image_path))
            save_path = os.path.join(save_dir, f"{filename_wo_extension}_aug_{i}.{extension}")
            augmented_image.save(save_path)
            return class_name
        
        except Exception as e:
            self.logger.error(f"Error augmented {image_path}: {e}.")
            return None


    def _balance_minority_class(self) -> None:
        tasks = []

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

                for i, image_path in enumerate(selected_paths, start=1):
                    tasks.append((image_path, save_dir, class_name, i))

        if self.cfg.use_multiprocessing:
            with Pool(processes=cpu_count()) as pool:
                results = list(tqdm(pool.imap(self._process_augmentation, tasks), total=len(tasks), desc="Augmented Images"))

        else:
            results = [self._process_augmentation(task) for task in tqdm(tasks, desc="Augmented Images")]

        class_image_count = Counter(filter(None, results))
        self.logger.info(f"Augmentation completed: {dict(class_image_count)}.\n")

    def run_pipeline(self) -> None:
        """Runs the entire image processing pipeline: distribution, directory setup, copying, and augmentation."""
        start_time = time.time()

        self._get_class_distribution()
        self._augment_and_save()
        self._balance_minority_class()

        end_time = time.time()
        elapsed_time = end_time - start_time
        self.logger.info(f"Data processing took {elapsed_time:.6f} seconds.\n")