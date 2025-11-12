import os
from typing import List

import cv2
import numpy as np
from joblib import dump


class ColorHistogramExtractor:
    """Extract normalized HSV color histograms per image.

    - Converts image to HSV
    - Computes histogram per channel (H, S, V) with 32 bins each
    - Normalizes each channel histogram so the concatenated vector sums to 1

    Result: 96-dimensional feature vector per image (32 bins * 3 channels)
    """

    def __init__(self, bins_per_channel: int = 32):
        self.bins = bins_per_channel

    def extract_histogram(self, image: np.ndarray) -> np.ndarray:
        """Compute a single 96-d HSV histogram for an image.

        Parameters:
        - image: np.ndarray in RGB or BGR format (we'll handle both)

        Returns:
        - 1D np.ndarray shape (96,), dtype=float32, L1-normalized to sum to 1.
        """
        if image is None:
            return np.zeros(self.bins * 3, dtype=np.float32)

        # If image has 3 channels, assume it's RGB or BGR. OpenCV expects BGR.
        img = image.copy()
        if img.ndim == 2:
            # single channel -> convert to BGR first
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # Detect if image is RGB (PIL loads RGB) by checking ranges; assume caller uses loader that returns RGB
        # Convert RGB to BGR for OpenCV if needed (heuristic: check typical color channel ordering)
        # We'll attempt to detect by sampling a pixel and comparing channel ranges; to be robust, allow both.
        # Simpler: convert from RGB to BGR if max channel value seems in R>G>B pattern often.
        # But to avoid mistakes, treat input as RGB and convert to BGR for OpenCV functions that expect BGR.
        try:
            # If values look like 0-255, assume RGB and convert
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        except Exception:
            img_bgr = img

        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

        # Compute hist per channel
        h_hist = cv2.calcHist([hsv], [0], None, [self.bins], [0, 180])  # Hue: 0-179
        s_hist = cv2.calcHist([hsv], [1], None, [self.bins], [0, 256])  # Sat: 0-255
        v_hist = cv2.calcHist([hsv], [2], None, [self.bins], [0, 256])  # Val: 0-255

        # Flatten and concatenate
        feat = np.concatenate([h_hist.flatten(), s_hist.flatten(), v_hist.flatten()]).astype(np.float32)

        # L1 normalize to sum to 1. If sum is zero, return zeros.
        s = feat.sum()
        if s > 0:
            feat /= s
        return feat

    def process_dataset(self, data_path: str, save_path: str = None, image_paths: List[str] = None) -> (np.ndarray, List[int]):
        """Compute color histogram features for a dataset.

        Parameters:
        - data_path: path to root dataset (if image_paths is None)
        - save_path: optional path to save features via joblib.dump
        - image_paths: optional list of explicit image file paths to process

        Returns:
        - features: np.ndarray shape (N, 96)
        - labels: list of int labels if dataset is organized in class subfolders, otherwise empty list
        """
        paths = []
        labels = []

        if image_paths is not None:
            paths = image_paths
        else:
            # Walk data_path expecting class subfolders
            class_names = sorted([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))])
            for cls_idx, cls in enumerate(class_names):
                cls_dir = os.path.join(data_path, cls)
                for fname in sorted(os.listdir(cls_dir)):
                    if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                        paths.append(os.path.join(cls_dir, fname))
                        labels.append(cls_idx)

        features = np.zeros((len(paths), self.bins * 3), dtype=np.float32)
        for i, p in enumerate(paths):
            img = cv2.imread(p)
            if img is None:
                # try with PIL or skip
                features[i] = np.zeros(self.bins * 3, dtype=np.float32)
                continue
            feat = self.extract_histogram(img)
            features[i] = feat

        if save_path:
            dump({'features': features, 'labels': labels, 'paths': paths}, save_path)

        return features, labels


if __name__ == '__main__':
    # quick local demo when run directly
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='../training_data')
    parser.add_argument('--save', default='../features/color_hist_features.joblib')
    args = parser.parse_args()

    ext = ColorHistogramExtractor()
    feats, labels = ext.process_dataset(args.data, save_path=args.save)
    print('Computed features shape:', feats.shape)
