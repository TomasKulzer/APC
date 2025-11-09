import os
from PIL import Image
import numpy as np

class ImageLoader:
    def __init__(self, root_dir, image_size=(224, 224)):
        """
        Initialize the image loader.

        Parameters:
        - root_dir: Directory containing subfolders per class label.
        - image_size: Desired image size as a tuple (width, height).
        """
        self.root_dir = root_dir
        self.image_size = image_size
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
        self.class_names = sorted(entry.name for entry in os.scandir(self.root_dir) if entry.is_dir())
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
        img = img.resize(self.image_size, Image.ANTIALIAS)
        return np.array(img)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Get the resized image and label at index `idx`.
        """
        image = self.load_image(self.image_paths[idx])
        label = self.labels[idx]
        return image, label
