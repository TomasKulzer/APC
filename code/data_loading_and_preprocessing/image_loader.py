import os
from PIL import Image
import numpy as np
import cv2
import random
from typing import Tuple

class ImageLoader:
    def __init__(self, root_dir, image_size=(224, 224), *, mode: str = 'train', augment: bool = False,
                 rotation_deg: int = 15, flip_prob: float = 0.5, color_jitter_prob: float = 0.5,
                 noise_prob: float = 0.1):
        """
        Initialize the image loader.

        Parameters:
        - root_dir: Directory containing subfolders per class label.
        - image_size: Desired image size as a tuple (width, height).
        """
        self.root_dir = root_dir
        self.image_size = image_size
        self.mode = mode
        self.augment = augment and (mode == 'train')
        self.rotation_deg = rotation_deg
        self.flip_prob = flip_prob
        self.color_jitter_prob = color_jitter_prob
        self.noise_prob = noise_prob
        self.class_names = []
        self.class_to_idx = {}
        self.image_paths = []
        self.labels = []

        self._prepare_dataset()

    def _prepare_dataset(self):
        """
        Scan the root directory for class subfolders,
        map class names to ordinal labels,
        collect image paths and labels.
        """
        """self.class_names = sorted(entry.name for entry in os.scandir(self.root_dir) if entry.is_dir())"""
        self.class_names = ["resistor", "capacitor", "transistor", "IC"]

        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.class_names)}

        for cls_name in self.class_names:
            cls_dir = os.path.join(self.root_dir, cls_name)
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    self.image_paths.append(os.path.join(cls_dir, fname))
                    self.labels.append(self.class_to_idx[cls_name])

    def load_image(self, image_path):
        """
        Load an image file, resize it, and convert to numpy array.

        Returns:
        - image as numpy array of shape (height, width, channels)
        """
        img = Image.open(image_path).convert('RGB')
        img = img.resize(self.image_size, resample=Image.Resampling.LANCZOS)
        return np.array(img)

    def _apply_augmentations(self, image: np.ndarray) -> np.ndarray:
        """
        Apply a set of random augmentations to the input image (uint8 HxWx3).
        Uses OpenCV for geometric transforms and simple pixel-wise operations.
        """
        img = image.copy()
        h, w = img.shape[:2]

        # Random rotation
        if self.rotation_deg > 0:
            angle = random.uniform(-self.rotation_deg, self.rotation_deg)
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)

        # Random horizontal flip
        if random.random() < self.flip_prob:
            img = cv2.flip(img, 1)

        # Color jitter / brightness & contrast
        if random.random() < self.color_jitter_prob:
            # Contrast (alpha) and brightness (beta)
            alpha = random.uniform(0.9, 1.1)  # contrast
            beta = random.uniform(-20, 20)    # brightness
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

            # Small saturation jitter in HSV space
            try:
                hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
                sat_scale = random.uniform(0.9, 1.1)
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_scale, 0, 255)
                img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
            except Exception:
                # Fallback: ignore HSV step if conversion fails
                pass

        # Gaussian noise
        if random.random() < self.noise_prob:
            sigma = random.uniform(1.0, 5.0)
            noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        return img

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Get the resized image and label at index `idx`.
        """
        image = self.load_image(self.image_paths[idx])
        # Apply on-the-fly augmentations only for training
        if self.augment:
            image = self._apply_augmentations(image)
        label = self.labels[idx]
        return image, label
