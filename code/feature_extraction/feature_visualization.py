import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure
from skimage.feature import hog
import seaborn as sns
from data_loading_and_preprocessing.image_loader import ImageLoader
from feature_extraction.hog_extractor import HOGFeatureExtractor

class FeatureVisualizer:
    @staticmethod
    def visualize_hog_features(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
        """
        Visualize HOG features overlaid on the original image
        
        Parameters:
        - image: Input image
        - orientations, pixels_per_cell, cells_per_block: HOG parameters
        """
        # Convert RGB to grayscale if needed
        if len(image.shape) == 3:
            from skimage.color import rgb2gray
            image_gray = rgb2gray(image)
        else:
            image_gray = image

        # Get HOG features and visualization
        features, hog_image = hog(
            image_gray,
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            block_norm='L2-Hys',
            visualize=True
        )

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Plot original image
        ax1.imshow(image)
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        # Plot HOG visualization
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
        ax2.imshow(hog_image_rescaled, cmap='hot')
        ax2.set_title('HOG Feature Visualization')
        ax2.axis('off')
        
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_feature_heatmap(features, sample_idx=0, shape=(224, 224)):
        """
        Plot a heatmap of the HOG feature vector for a single sample
        
        Parameters:
        - features: Array of HOG features
        - sample_idx: Index of the sample to visualize
        - shape: Original image shape for reference
        """
        # Get features for the specified sample
        feature_vector = features[sample_idx]
        
        # Reshape to make it more visual (approximate spatial representation)
        feature_dim = int(np.sqrt(len(feature_vector)))
        feature_map = feature_vector[:feature_dim**2].reshape(feature_dim, feature_dim)
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(feature_map, cmap='viridis')
        plt.title(f'HOG Feature Heatmap (Sample {sample_idx})')
        plt.xlabel('Feature Dimension')
        plt.ylabel('Feature Dimension')
        
        return plt.gcf()

    @staticmethod
    def plot_class_average_features(features, labels, class_names):
        """
        Plot average HOG features for each class
        
        Parameters:
        - features: Array of HOG features
        - labels: Array of corresponding labels
        - class_names: List of class names
        """
        n_classes = len(class_names)
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        axes = axes.ravel()
        
        for i, class_name in enumerate(class_names):
            # Get features for current class
            class_features = features[labels == i]
            avg_features = np.mean(class_features, axis=0)
            
            # Reshape to make it more visual
            feature_dim = int(np.sqrt(len(avg_features)))
            feature_map = avg_features[:feature_dim**2].reshape(feature_dim, feature_dim)
            
            # Plot
            sns.heatmap(feature_map, cmap='viridis', ax=axes[i])
            axes[i].set_title(f'Average HOG Features: {class_name}')
            axes[i].set_xlabel('Feature Dimension')
            axes[i].set_ylabel('Feature Dimension')
        
        plt.tight_layout()
        return fig

def visualize_sample_with_features(data_path, save_path):
    """
    Create and save visualizations for a sample image
    
    Parameters:
    - data_path: Path to the training data
    - save_path: Path to the saved HOG features
    """
    # Load data
    loader = ImageLoader(data_path)
    features, labels, class_names = HOGFeatureExtractor.load_features(save_path)
    
    # Get a sample image
    image, label = loader[0]
    
    # Create visualizations
    vis = FeatureVisualizer()
    
    # 1. HOG visualization
    fig1 = vis.visualize_hog_features(image)
    fig1.savefig('hog_visualization.png')
    
    # 2. Feature heatmap
    fig2 = vis.plot_feature_heatmap(features)
    fig2.savefig('feature_heatmap.png')
    
    # 3. Class average features
    fig3 = vis.plot_class_average_features(features, labels, class_names)
    fig3.savefig('class_average_features.png')
    
    print("Visualizations saved as:")
    print("- hog_visualization.png")
    print("- feature_heatmap.png")
    print("- class_average_features.png")