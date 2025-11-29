#!/usr/bin/env python3
"""
05 - Visualize HOG features and class averages

Usage:
    python code/05_visualize.py --data dataset --features ../features/hog_features.joblib

This saves `hog_visualization.png`, `feature_heatmap.png`, and `class_average_features.png` in the current working directory.
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(__file__))

from feature_extraction.feature_visualization import visualize_sample_with_features


def main(data_path: str, features_path: str):
    visualize_sample_with_features(data_path, features_path)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='dataset')
    p.add_argument('--features', default='../features/hog_features.joblib')
    args = p.parse_args()
    main(args.data, args.features)
