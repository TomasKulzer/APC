import os
import argparse

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

from data_loading_and_preprocessing.image_loader import ImageLoader
from feature_extraction.color_histogram import ColorHistogramExtractor


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def plot_hsv_histogram(hist, bins, save_path=None, title=None):
    """Plot H, S, V histograms (hist is 96-d or 3 x bins) and optionally save."""
    # hist may be concatenated 3*bins
    if hist.ndim != 1 or hist.size != 3 * bins:
        raise ValueError('hist must be 1D array of length 3*bins')

    h = hist[0:bins]
    s = hist[bins:2*bins]
    v = hist[2*bins:3*bins]

    fig, axes = plt.subplots(3, 1, figsize=(6, 6), constrained_layout=True)

    # Hue: color bars according to hue
    bin_edges_h = np.linspace(0, 180, bins + 1)
    bin_centers_h = 0.5 * (bin_edges_h[:-1] + bin_edges_h[1:])
    # Normalize to 0-1 for HSV colormap
    hsv_vals = np.stack([bin_centers_h / 180.0, np.ones_like(bin_centers_h), np.ones_like(bin_centers_h)], axis=1)
    colors_h = hsv_to_rgb(hsv_vals)

    axes[0].bar(np.arange(bins), h, color=colors_h, align='center')
    axes[0].set_title('Hue histogram')
    axes[0].set_xlim(-0.5, bins - 0.5)

    # Saturation
    axes[1].bar(np.arange(bins), s, color='tab:orange')
    axes[1].set_title('Saturation histogram')
    axes[1].set_xlim(-0.5, bins - 0.5)

    # Value
    axes[2].bar(np.arange(bins), v, color='tab:gray')
    axes[2].set_title('Value histogram')
    axes[2].set_xlim(-0.5, bins - 0.5)

    if title:
        fig.suptitle(title)

    if save_path:
        plt.savefig(save_path)
        plt.close(fig)
    else:
        plt.show()


def visualize_image_hist(image_path, extractor: ColorHistogramExtractor, save_dir: str, idx: int, bins: int):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f'Cannot read image: {image_path}')

    # Convert to RGB for display
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    hist = extractor.extract_histogram(img_bgr)

    ensure_dir(save_dir)
    # Create a combined figure: left image, right histogram
    fig = plt.figure(figsize=(10, 5), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1])

    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(img_rgb)
    ax_img.axis('off')
    ax_img.set_title(os.path.basename(image_path))

    ax_hist = fig.add_subplot(gs[0, 1])
    # Instead of reusing plot_hsv_histogram, draw inline to align with the figure
    h = hist[0:bins]
    s = hist[bins:2*bins]
    v = hist[2*bins:3*bins]

    x = np.arange(bins)
    # Hue colors
    bin_edges_h = np.linspace(0, 180, bins + 1)
    bin_centers_h = 0.5 * (bin_edges_h[:-1] + bin_edges_h[1:])
    hsv_vals = np.stack([bin_centers_h / 180.0, np.ones_like(bin_centers_h), np.ones_like(bin_centers_h)], axis=1)
    colors_h = hsv_to_rgb(hsv_vals)

    # Plot stacked subplots in the right axis area
    # We'll create inset axes for each channel
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    ax_h = inset_axes(ax_hist, width='100%', height='30%', bbox_to_anchor=(0, 0.66, 1, 0.33), bbox_transform=ax_hist.transAxes)
    ax_s = inset_axes(ax_hist, width='100%', height='30%', bbox_to_anchor=(0, 0.33, 1, 0.33), bbox_transform=ax_hist.transAxes)
    ax_v = inset_axes(ax_hist, width='100%', height='30%', bbox_to_anchor=(0, 0.0, 1, 0.33), bbox_transform=ax_hist.transAxes)

    ax_h.bar(x, h, color=colors_h, align='center')
    ax_h.set_xticks([])
    ax_h.set_ylabel('H')

    ax_s.bar(x, s, color='tab:orange', align='center')
    ax_s.set_xticks([])
    ax_s.set_ylabel('S')

    ax_v.bar(x, v, color='tab:gray', align='center')
    ax_v.set_ylabel('V')
    ax_v.set_xlabel('Bin')

    ax_hist.axis('off')

    out_path = os.path.join(save_dir, f'color_hist_vis_{idx}.png')
    plt.savefig(out_path)
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(description='Visualize color histograms for images')
    parser.add_argument('--data-dir', default='../training_data')
    parser.add_argument('--n-images', type=int, default=5)
    parser.add_argument('--save-dir', default='../features/color_hist_vis')
    parser.add_argument('--bins', type=int, default=32)
    args = parser.parse_args()

    loader = ImageLoader(args.data_dir, image_size=(224, 224))
    if len(loader.image_paths) == 0:
        print('No images found in', args.data_dir)
        return

    extractor = ColorHistogramExtractor(bins_per_channel=args.bins)

    ensure_dir(args.save_dir)
    selected = loader.image_paths[: args.n_images]
    for i, p in enumerate(selected):
        try:
            out = visualize_image_hist(p, extractor, args.save_dir, i, args.bins)
            print('Saved visualization:', out)
        except Exception as e:
            print(f'Failed for {p}: {e}')


if __name__ == '__main__':
    main()
