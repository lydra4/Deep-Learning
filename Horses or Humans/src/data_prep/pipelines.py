import logging
import os
import random
import shutil
import time
from collections import Counter
from multiprocessing import Pool, cpu_count
from typing import List, Optional

import omegaconf
from PIL import Image
from torchvision import datasets, transforms
from tqdm import tqdm


class ImagePipeline:
    """
    A class to process image datasets, including class distribution calculation, image copying,
    and data augmentation for balancing the dataset.

    This class performs the following tasks:
    1. Calculates class distributions.
    2. Creates necessary directories for processed data.
    3. Copies original images to test and validation directories.
    4. Performs data augmentation on the training dataset to balance the classes.

    Attributes:
        cfg (omegaconf.DictConfig): Configuration dictionary containing paths and augmentation settings.
        logger (logging.Logger): Logger to track the progress and any issues.
        dataset (torchvision.datasets.ImageFolder): Dataset loaded using torchvision.
        class_counts (collections.Counter): Counts of images per class in the dataset.
        processed_path (Optional[str]): Path where processed images will be stored.
        used_images (Optional[dict[str, List[str]]]): A dictionary to track images already used for test/val.
    """

    def __init__(
        self, cfg: omegaconf.DictConfig, logger: Optional[logging.Logger]
    ) -> None:
        """
        Initializes the ImagePipeline with configuration and logging.

        Args:
            cfg (omegaconf.DictConfig): Configuration dictionary containing paths and augmentation settings.
            logger (Optional[logging.Logger]): Logger for tracking progress. Defaults to None.
        """
        self.cfg = cfg
        self.logger = logger or logging.getLogger(__name__)
        self.dataset = datasets.ImageFolder(root=self.cfg.path_to_unprocessed_data)
        self.class_counts = Counter(
            [self.dataset.classes[label] for _, label in self.dataset.samples]
        )
        self.processed_path: Optional[str] = self.cfg.path_to_processed_data
        self.used_images: Optional[dict[str, List[str]]] = None

    def _get_class_distribution(self) -> None:
        """
        Logs the distribution of images across different classes in the dataset.

        This method calculates and logs the number of images per class in the dataset.
        """
        self.logger.info(f"Class distribution: {self.class_counts}.\n")

    def _copy_single_image(self, args):
        image_name, source_dir, destination_dir = args

        try:
            source = os.path.join(source_dir, image_name)
            destination = os.path.join(destination_dir, image_name)
            shutil.copy(source, destination)
        except Exception as e:
            self.logger.error(f"Failed to copy {image_name}: {e}.\n")

    def _copy_test_val_images(self):
        """
        Copies a specified number of test and validation images to their respective directories.

        This method selects random images from the source dataset and copies them into separate
        test and validation directories. It also tracks which images have been used to avoid duplication.
        """
        if self.used_images is None:
            self.used_images = {}

        for split in ["test", "val"]:
            tasks = []

            for class_name in self.dataset.classes:
                source_dir = os.path.join(self.cfg.path_to_unprocessed_data, class_name)
                destination_dir = os.path.join(
                    self.cfg.path_to_processed_data, split, class_name
                )
                os.makedirs(name=destination_dir, exist_ok=True)

                all_images = os.listdir(source_dir)
                selected_images = random.sample(
                    population=all_images, k=self.cfg.test_val_images
                )

                self.used_images.setdefault(class_name, set())
                self.used_images[class_name].update(selected_images)

                for image_name in selected_images:
                    tasks.append((image_name, source_dir, destination_dir))

            if self.cfg.use_multiprocessing:
                self.logger.info(
                    f"\nUsing {str(cpu_count())} processors to copy images into {split}.\n"
                )
                with Pool(cpu_count()) as pool:
                    list(
                        tqdm(
                            pool.imap(self._copy_single_image, tasks), total=len(tasks)
                        )
                    )

            else:
                self.logger.info(
                    f"\nCopying images into {split}, {class_name} on a single processor.\n"
                )
                for task in tqdm(tasks):
                    self._copy_single_image(task)

    def _data_augmentation(self, image: Image.Image) -> Image.Image:
        """
        Applies data augmentation to the input image.

        This method performs a series of random transformations such as color jitter, Gaussian blur,
        random resized crop, and random horizontal flip to augment the input image.

        Args:
            image (PIL.Image.Image): The image to be augmented.

        Returns:
            PIL.Image.Image: The augmented image.
        """
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
                transforms.RandomHorizontalFlip(p=self.cfg.random_horizontal_flip),
                transforms.ToTensor(),
                transforms.ToPILImage(),
            ]
        )

        return augment_transforms(img=image)

    def _process_augmentation(self, args) -> Optional[str]:
        """
        Processes and saves augmented images.

        This method takes an image, applies augmentation, and saves the resulting image to the destination
        directory.

        Args:
            args (tuple): Contains image path, save directory, class name, and index for naming.

        Returns:
            Optional[str]: The class name of the processed image or None in case of failure.
        """
        image_path, save_dir, class_name, i = args

        try:
            original_image = Image.open(image_path).convert("RGB")
            augmented_image = self._data_augmentation(image=original_image)

            filename_wo_extension, extension = os.path.splitext(
                os.path.basename(image_path)
            )
            save_path = os.path.join(
                save_dir, f"{filename_wo_extension}_aug_{i}.{extension}"
            )
            augmented_image.save(save_path)
            return class_name

        except Exception as e:
            self.logger.error(f"Error augmented {image_path}: {e}.")
            return None

    def _balance_and_augment_train(self) -> None:
        """
        Balances and augments the training dataset by duplicating minority class images.

        This method ensures that each class has the same number of images by augmenting the images
        from the minority classes and copying them into the training dataset.

        This operation can be parallelized using multiple processors.

        Args:
            None
        """
        tasks = []
        target_count = self.cfg.train_images

        for class_name in self.dataset.classes:
            source_dir = os.path.join(self.cfg.path_to_unprocessed_data, class_name)
            destination_dir = os.path.join(
                self.cfg.path_to_processed_data, "train", class_name
            )
            os.makedirs(name=destination_dir, exist_ok=True)

            all_images = set(os.listdir(source_dir))
            used_images = self.used_images.get(class_name, set())
            available_images = list(all_images - used_images)

            for i, image_name in enumerate(available_images):
                image_path = os.path.join(source_dir, image_name)
                tasks.append((image_path, destination_dir, class_name, i))

            num_to_augment = target_count - len(available_images)
            if num_to_augment > 0:
                selected_paths = random.choices(
                    population=available_images, k=num_to_augment
                )
                start_index = len(available_images)

                for i, image_name in enumerate(selected_paths):
                    image_path = os.path.join(source_dir, image_name)
                    tasks.append(
                        (image_path, destination_dir, class_name, start_index + i)
                    )

        if self.cfg.use_multiprocessing:
            self.logger.info(
                f"\nUsing {str(cpu_count())} processors to augment train.\n"
            )
            with Pool(cpu_count()) as pool:
                list(
                    tqdm(
                        pool.imap(self._process_augmentation, tasks),
                        total=len(tasks),
                        desc="Augmenting Train",
                    )
                )
        else:
            self.logger.info("\nUsing single processor to augment train.\n")
            for task in tqdm(tasks, desc="Augmenting Train"):
                self._process_augmentation(task)

    def run_pipeline(self) -> None:
        """
        Executes the complete image processing pipeline.

        This method runs the entire pipeline which includes getting class distribution,
        copying test and validation images, and augmenting the training dataset.

        The elapsed time for the process is logged at the end.

        Args:
            None
        """
        start_time = time.time()

        self._get_class_distribution()
        self._copy_test_val_images()
        self._balance_and_augment_train()

        end_time = time.time()
        elapsed_time = end_time - start_time

        if self.cfg.use_multiprocessing:
            self.logger.info(
                f"\nData processing took {elapsed_time:.6f} seconds using multiprocessing.\n"
            )
        else:
            self.logger.info(
                f"\nData processing took {elapsed_time:.6f} seconds on a single processor.\n"
            )
