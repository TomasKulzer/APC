import os
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt

from data_loading_and_preprocessing.image_loader import ImageLoader
from feature_extraction.sift import SIFTBagOfWords


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def visualize_keypoints_and_bow(image_path, sift_bow, save_dir, idx=0):
    """Draw SIFT keypoints on the image and plot its BoW histogram.

    Returns paths to saved images (keypoints image, histogram).
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = sift_bow.sift.detectAndCompute(gray, None)

    # Draw keypoints (use rich drawing)
    img_kp = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    ensure_dir(save_dir)
    kp_path = os.path.join(save_dir, f"sift_keypoints_{idx}.png")
    cv2.imwrite(kp_path, img_kp)

    hist_path = None
    if sift_bow.kmeans is not None and descriptors is not None and descriptors.shape[0] > 0:
        hist = sift_bow.get_image_bow(descriptors)
        # Plot histogram
        plt.figure(figsize=(8, 3))
        plt.bar(np.arange(len(hist)), hist)
        plt.xlabel('Visual word index')
        plt.ylabel('L2-normalized frequency')
        plt.title(f'BoW histogram for {os.path.basename(image_path)}')
        hist_path = os.path.join(save_dir, f"sift_bow_hist_{idx}.png")
        plt.tight_layout()
        plt.savefig(hist_path)
        plt.close()

    return kp_path, hist_path


def main():
    parser = argparse.ArgumentParser(description='Visualize SIFT keypoints and BoW histograms')
    parser.add_argument('--data-dir', default='../training_data', help='Path to training data root')
    parser.add_argument('--k', type=int, default=128, help='Vocabulary size (k)')
    parser.add_argument('--n-images', type=int, default=5, help='Number of images to visualize (from start)')
    parser.add_argument('--save-dir', default='../features/sift_vis', help='Directory to save visualizations')
    parser.add_argument('--max-sample-desc', type=int, default=20000, help='Max descriptors to sample for vocab')
    args = parser.parse_args()

    loader = ImageLoader(args.data_dir, image_size=(224, 224))
    if len(loader.image_paths) == 0:
        print('No images found in', args.data_dir)
        return

    sift_bow = SIFTBagOfWords(k=args.k)

    # Collect descriptors from a subset (or all) images
    all_desc = []
    chosen = loader.image_paths[: args.n_images]
    for p in chosen:
        img = loader.load_image(p)
        # convert to gray for SIFT
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
        _, desc = sift_bow.sift.detectAndCompute(gray, None)
        if desc is not None and desc.shape[0] > 0:
            all_desc.append(desc)

    if all_desc:
        all_desc = np.vstack(all_desc)
    else:
        all_desc = np.empty((0, 128), dtype=np.float32)

    # Fit vocabulary (this uses the MiniBatchKMeans-based fitter in sift.py)
    sift_bow.fit_vocab(all_desc, max_samples=args.max_sample_desc)

    ensure_dir(args.save_dir)
    print(f"Fitted vocabulary of size: {getattr(sift_bow, 'vocab', None).shape}")

    for i, p in enumerate(chosen):
        try:
            kp_path, hist_path = visualize_keypoints_and_bow(p, sift_bow, args.save_dir, idx=i)
            print('Saved keypoints:', kp_path)
            if hist_path:
                print('Saved BoW histogram:', hist_path)
            else:
                print('No BoW histogram (no descriptors or vocab).')
        except Exception as e:
            print(f'Failed to visualize {p}: {e}')


if __name__ == '__main__':
    main()
