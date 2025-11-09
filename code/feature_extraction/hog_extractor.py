import numpy as np
from skimage.feature import hog
from skimage.color import rgb2gray
import os
import joblib
from tqdm import tqdm
from data_loading_and_preprocessing.image_loader import ImageLoader

class HOGFeatureExtractor:
    def __init__(self, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
        """
        Initialize HOG Feature Extractor

        Parameters:
        - orientations: Number of orientation bins (default=9)
        - pixels_per_cell: Size of cell for HOG (default=(8,8))
        - cells_per_block: Number of cells in each block (default=(2,2))
        """
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block

    def extract_features(self, image):
        """
        Extract HOG features from a single image

        Parameters:
        - image: RGB image as numpy array (height, width, channels)

        Returns:
        - hog_features: 1D array of HOG features
        """
        # Convert to grayscale if image is RGB
        if len(image.shape) == 3:
            image = rgb2gray(image)

        # Extract HOG features
        hog_features = hog(
            image,
            orientations=self.orientations,
            pixels_per_cell=self.pixels_per_cell,
            cells_per_block=self.cells_per_block,
            block_norm='L2-Hys',
            feature_vector=True
        )

        return hog_features

    def process_dataset(self, data_path, save_path=None, batch_size=32):
        """
        Process entire dataset and extract HOG features

        Parameters:
        - data_path: Path to the dataset root directory
        - save_path: Optional path to save features to disk
        - batch_size: Number of images to process at once

        Returns:
        - features: Array of HOG features for all images
        - labels: Array of corresponding labels
        """
        # Initialize image loader
        loader = ImageLoader(data_path)
        total_images = len(loader)
        
        # Initialize arrays to store features and labels
        features = []
        labels = []

        # Process images in batches with progress bar
        for i in tqdm(range(0, total_images, batch_size), desc="Extracting HOG features"):
            batch_end = min(i + batch_size, total_images)
            
            # Process batch
            for idx in range(i, batch_end):
                image, label = loader[idx]
                hog_features = self.extract_features(image)
                features.append(hog_features)
                labels.append(label)

        # Convert to numpy arrays
        features = np.array(features)
        labels = np.array(labels)

        # Save to disk if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            joblib.dump({
                'features': features,
                'labels': labels,
                'class_names': loader.class_names
            }, save_path)
            print(f"Features saved to {save_path}")

        return features, labels

    @staticmethod
    def load_features(save_path):
        """
        Load previously saved features from disk

        Parameters:
        - save_path: Path to the saved features file

        Returns:
        - features: Array of HOG features
        - labels: Array of corresponding labels
        - class_names: List of class names
        """
        data = joblib.load(save_path)
        return data['features'], data['labels'], data['class_names']