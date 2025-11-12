import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize

class SIFTBagOfWords:
    def __init__(self, k=128):  # k: vocabulary size
        self.k = k
        self.sift = cv2.SIFT_create()
        self.kmeans = None
        self.vocab = None

    def extract_sift_descriptors(self, image):
        """Extract SIFT descriptors from an image."""
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        keypoints, descriptors = self.sift.detectAndCompute(image, None)
        return descriptors if descriptors is not None else np.empty((0,128), dtype=np.float32)

    def fit_vocab(self, all_descriptors, max_samples=200000, batch_size=1000):
        """Fit a (MiniBatch) KMeans vocabulary using descriptors.

        To avoid very long runs on huge descriptor sets, we sample up to
        `max_samples` descriptors uniformly at random and use MiniBatchKMeans
        for scalable clustering.
        """
        if all_descriptors is None or all_descriptors.shape[0] == 0:
            # Nothing to fit
            self.kmeans = None
            self.vocab = np.empty((0, all_descriptors.shape[1] if all_descriptors.size else 128))
            return

        n_desc = all_descriptors.shape[0]
        if n_desc > max_samples:
            rng = np.random.RandomState(42)
            idx = rng.choice(n_desc, max_samples, replace=False)
            sample = all_descriptors[idx]
        else:
            sample = all_descriptors

        # Use MiniBatchKMeans for speed and lower memory usage
        self.kmeans = MiniBatchKMeans(n_clusters=self.k, random_state=42, batch_size=batch_size)
        self.kmeans.fit(sample)
        self.vocab = self.kmeans.cluster_centers_

    def get_image_bow(self, descriptors):
        """Convert descriptors to BoW histogram (normalized)."""
        if descriptors is None or descriptors.shape[0] == 0:
            hist = np.zeros(self.k)
        else:
            words = self.kmeans.predict(descriptors)
            hist, _ = np.histogram(words, bins=np.arange(self.k+1))
        hist = normalize(hist.reshape(1, -1), norm='l2')[0]
        return hist

    def process_images(self, image_paths):
        """High-level pipeline for BoW extraction."""
        # Step 1: Collect all SIFT descriptors
        all_desc = []
        image_desc = []
        for path in image_paths:
            img = cv2.imread(path)
            desc = self.extract_sift_descriptors(img)
            image_desc.append(desc)
            if desc.shape[0] > 0:
                all_desc.append(desc)
        all_desc = np.vstack(all_desc) if all_desc else np.empty((0,128))
        # Step 2: Create vocabulary (fit KMeans)
        self.fit_vocab(all_desc)
        # Step 3: Generate BoW histograms for each image
        bows = np.array([self.get_image_bow(desc) for desc in image_desc])
        return bows
