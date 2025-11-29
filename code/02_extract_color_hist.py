#!/usr/bin/env python3
"""
02 - Extract color histogram features for each split (non-interactive)

This script reads `dataset/splits.joblib` and writes:
  - ../features/colorhist_train.joblib
  - ../features/colorhist_val.joblib
  - ../features/colorhist_test.joblib

Runs without CLI options; run from repo root:
    python code/02_extract_color_hist.py
"""
import os
import sys
import joblib
import numpy as np
import cv2
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from feature_extraction.color_histogram import ColorHistogramExtractor


def main():
    data_dir = 'dataset'
    splits_file = os.path.join(data_dir, 'splits.joblib')
    out_dir = 'features'
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(splits_file):
        raise FileNotFoundError(f"Splits file not found: {splits_file}. Run 01_prepare_data.py first.")

    splits = joblib.load(splits_file)
    extractor = ColorHistogramExtractor()

    for split in ('train', 'val', 'test'):
        rel_paths = splits[f'{split}_paths']
        labels = splits[f'{split}_labels']
        # Convert relative paths to full paths
        paths = []
        for p in rel_paths:
            if os.path.isabs(p) or p.startswith(data_dir + os.sep):
                paths.append(p)
            else:
                paths.append(os.path.join(data_dir, p))

        save_path = os.path.join(out_dir, f'colorhist_{split}.joblib')
        print(f"Extracting color hist for split '{split}' -> {save_path}")

        # Process with progress bar
        n = len(paths)
        feats = np.zeros((n, extractor.bins * 3), dtype=np.float32)
        lbls_out = []
        for i, p in enumerate(tqdm(paths, desc=f"ColorHist {split}")):
            img = cv2.imread(p)
            if img is None:
                # try to open with PIL as a fallback
                try:
                    from PIL import Image
                    pil_img = Image.open(p).convert('RGB')
                    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                except Exception:
                    feats[i] = np.zeros(extractor.bins * 3, dtype=np.float32)
                    lbls_out.append(labels[i])
                    continue
            feat = extractor.extract_histogram(img)
            feats[i] = feat
            lbls_out.append(labels[i])

        joblib.dump({'features': feats, 'labels': lbls_out, 'paths': paths}, save_path)
        print(f"Saved {feats.shape[0]} color-hist features for {split}")


if __name__ == '__main__':
    main()
