"""
Dataset Visualization Demo

This script creates a 5x5 grid visualization of random samples from the dataset.
Each row corresponds to one of the 5 ripeness classes.
"""

import os
import random
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

def create_dataset_grid(dataset_path='Dataset', output_path='visualizations/dataset_demo.png'):
    """
    Create a 5x5 grid of random dataset samples.
    
    Args:
        dataset_path: Path to the dataset directory
        output_path: Where to save the output image
    """
    # Class directories
    classes = [
        '0Immature',
        '1PartiallyRipe',
        '2FullyRipe',
        '3OverRipe',
        '4Decayed'
    ]
    
    class_names = [
        'Immature',
        'Partially Ripe',
        'Fully Ripe',
        'Over Ripe',
        'Decayed'
    ]
    
    # Create figure with 5x5 grid
    fig, axes = plt.subplots(5, 5, figsize=(15, 15))
    
    # For each class (row)
    for row_idx, (class_dir, class_name) in enumerate(zip(classes, class_names)):
        class_path = Path(dataset_path) / class_dir
        
        # Get all images in this class
        image_files = list(class_path.glob('*.jpg')) + list(class_path.glob('*.png'))
        
        if len(image_files) == 0:
            print(f"Warning: No images found in {class_path}")
            continue
        
        # Select 5 random images
        selected_images = random.sample(image_files, min(5, len(image_files)))
        
        # Display images in this row
        for col_idx, img_path in enumerate(selected_images):
            ax = axes[row_idx, col_idx]
            
            # Load and display image
            img = Image.open(img_path)
            ax.imshow(img)
            ax.axis('off')
            
            # Add class label to first column
            if col_idx == 0:
                ax.text(-0.1, 0.5, class_name, transform=ax.transAxes,
                       fontsize=14, fontweight='bold', rotation=90,
                       verticalalignment='center', horizontalalignment='right')
    
    #plt.suptitle('Dataset Sample Grid - 5 Classes x 5 Samples', fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Create output directory if needed
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Dataset grid saved to: {output_path}")


def main():
    print("="*60)
    print("DATASET VISUALIZATION DEMO")
    print("="*60)
    
    create_dataset_grid()
    
    print("\n" + "="*60)
    print("DEMO COMPLETED")
    print("="*60)


if __name__ == '__main__':
    main()
