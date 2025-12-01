#!/usr/bin/env python3
"""
02 - Extract SIFT Bag-of-Words features per split (non-interactive)

This script reads `dataset/splits.joblib` and writes:
  - ../features/sift_train.joblib
  - ../features/sift_val.joblib
  - ../features/sift_test.joblib

Note: SIFT requires OpenCV (opencv-contrib-python for SIFT in some builds).
Run from repo root:
    python code/02_extract_sift.py
"""
import os
import sys
import joblib
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(__file__))
from feature_extraction.sift import SIFTBagOfWords


def main():
    data_dir = 'dataset'
    splits_file = os.path.join(data_dir, 'splits.joblib')
    out_dir = 'features'
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(splits_file):
        raise FileNotFoundError(f"Splits file not found: {splits_file}. Run 01_prepare_data.py first.")

    splits = joblib.load(splits_file)
    bow = SIFTBagOfWords(k=128)

    # To build the vocabulary robustly, we'll collect descriptors from the training split only
    train_rel = splits['train_paths']
    train_paths = []
    for p in train_rel:
        if os.path.isabs(p) or p.startswith(data_dir + os.sep):
            train_paths.append(p)
        else:
            train_paths.append(os.path.join(data_dir, p))

    from tqdm import tqdm
    
    print('Collecting descriptors from training images to fit vocabulary...')
    all_desc = []
    for p in tqdm(train_paths, desc='Collecting SIFT descriptors'):
        img = cv2.imread(p)
        desc = bow.extract_sift_descriptors(img)
        if desc is not None and desc.shape[0] > 0:
            all_desc.append(desc)
    if all_desc:
        all_desc = np.vstack(all_desc)
    else:
        all_desc = np.empty((0, 128), dtype=np.float32)

    print(f'Fitting SIFT vocabulary on {all_desc.shape[0]} descriptors (this may take some time)')
    bow.fit_vocab(all_desc)

    # Now process each split to produce BoW histograms
    for split in ('train', 'val', 'test'):
        rel_paths = splits[f'{split}_paths']
        labels = splits[f'{split}_labels']
        paths = []
        for p in rel_paths:
            if os.path.isabs(p) or p.startswith(data_dir + os.sep):
                paths.append(p)
            else:
                paths.append(os.path.join(data_dir, p))
        
        print(f'Processing {len(paths)} images for split {split}...')
        
        # Use process_images-like flow but with pre-fit vocabulary
        image_descs = []
        for p in tqdm(paths, desc=f'SIFT extract {split}'):
            img = cv2.imread(p)
            desc = bow.extract_sift_descriptors(img)
            image_descs.append(desc)
        # Convert descriptors to histograms
        bows = np.array([bow.get_image_bow(desc) for desc in image_descs])
        save_path = os.path.join(out_dir, f'sift_{split}.joblib')
        joblib.dump({'features': bows, 'labels': labels, 'paths': paths}, save_path)
        print(f'Saved sift BoW features to {save_path} (shape {bows.shape})')


if __name__ == '__main__':
    main()
