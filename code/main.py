from data_loading_and_preprocessing.image_loader import ImageLoader


if __name__ == "__main__":
    loader = ImageLoader('../training_data', image_size=(224,224))
    print(f"Found classes: {loader.class_names}")
    print(f"Number of samples: {len(loader)}")
    image, label = loader[0]
    print(f"Image shape: {image.shape}, Label: {label}")
