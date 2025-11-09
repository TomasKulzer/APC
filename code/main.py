from data_loading_and_preprocessing.image_loader import ImageLoader
from data_loading_and_preprocessing.ordinal_encoding import OrdinalEncoder
from feature_extraction.hog_extractor import HOGFeatureExtractor
from feature_extraction.feature_visualization import visualize_sample_with_features


if __name__ == "__main__":
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