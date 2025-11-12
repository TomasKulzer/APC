from data_loading_and_preprocessing.image_loader import ImageLoader
from data_loading_and_preprocessing.ordinal_encoding import OrdinalEncoder
from feature_extraction.hog_extractor import HOGFeatureExtractor
from feature_extraction.feature_visualization import visualize_sample_with_features
from feature_extraction.sift import SIFTBagOfWords
from feature_extraction.color_histogram import ColorHistogramExtractor
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Feature extraction pipeline')
    parser.add_argument('--extract-color-hist', action='store_true', help='Also extract HSV color histograms (96-d)')
    parser.add_argument('--color-save', default='../features/color_hist_features.joblib', help='Where to save color histogram features')
    args = parser.parse_args()

    loader = ImageLoader('../training_data', image_size=(224,224))
    print(f"Found classes: {loader.class_names}")
    print(f"Number of samples: {len(loader)}")
    image, label = loader[0]
    print(f"Image shape: {image.shape}, Label: {label}")
    encoder = OrdinalEncoder(["resistor", "capacitor", "transistor", "IC"])
    y = ["resistor", "IC", "capacitor", "transistor"]  # Can be integers as well
    ord_labels = encoder.encode(y)
    print(ord_labels)

    # Initialize HOG extractor
    hog_extractor = HOGFeatureExtractor()

    # Process entire dataset
    features, labels = hog_extractor.process_dataset(
    data_path='../training_data',
    save_path='../features/hog_features.joblib'
    )

    print(f"Extracted features shape: {features.shape}")
    print(f"Number of samples: {len(labels)}")
    
    # Generate and save visualizations
    visualize_sample_with_features('../training_data', '../features/hog_features.joblib')
    
    # Initialize and run SIFT Bag-of-Words (if you have images available)
    sift_bow = SIFTBagOfWords(k=128)
    # Use the image paths gathered by the ImageLoader as a safe default
    image_paths = loader.image_paths
    if len(image_paths) > 0:
        bow_features = sift_bow.process_images(image_paths)
        print(f"BoW features shape: {bow_features.shape}")
    else:
        print("No image paths found for SIFT BoW processing; skipping.")
    # bow_features.shape == (num_images, 128)

    # Optionally extract color histograms
    if args.extract_color_hist:
        print('Running ColorHistogramExtractor...')
        color_ext = ColorHistogramExtractor(bins_per_channel=32)
        # use loader.image_paths and let the extractor save to the provided path
        color_features, color_labels = color_ext.process_dataset(None, save_path=args.color_save, image_paths=loader.image_paths)
        print(f'Color histogram features shape: {color_features.shape}')
        print(f'Color features saved to: {args.color_save}')
