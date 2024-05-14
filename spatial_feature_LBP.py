import numpy as np
import matplotlib.pyplot as plt
import cv2

METHOD = 'uniform'
plt.rcParams['font.size'] = 9


def plot_circle(ax, center, radius, color):
    circle = plt.Circle(center, radius, facecolor=color, edgecolor='0.5')
    ax.add_patch(circle)


def plot_lbp_model(ax, binary_values):
    """Draw the schematic for a local binary pattern."""
    # Geometry spec
    theta = np.deg2rad(360/len(binary_values))
    R = 8
    r = 0.15
    w = R+1.5
    gray = '0.5'

    # Draw the central pixel.
    plot_circle(ax, (0, 0), radius=r, color=gray)
    # Draw the surrounding pixels.
    for i, facecolor in enumerate(binary_values):
        x = R * np.cos(i * theta)
        y = R * np.sin(i * theta)
        plot_circle(ax, (x, y), radius=r, color=str(facecolor))

    # Draw the pixel grid.
    for x in np.linspace(-w, w, 4):
        ax.axvline(x, color=gray)
        ax.axhline(x, color=gray)

    # Tweak the layout.
    ax.axis('image')
    #ax.axis('off')
    size = w + 0.2
    ax.set_xlim(-size, size)
    ax.set_ylim(-size, size)

def plot_lbp_location():
    #fig, axes = plt.subplots(ncols=5, figsize=(7, 2))
    fig, axes = plt.subplots(ncols=2, figsize=(7, 2))

    #titles = ['flat', 'flat', 'edge', 'corner', 'non-uniform']
    titles = ['flat']

    binary_patterns = [np.zeros(58),
                    #   np.ones(8),
                    #   np.hstack([np.ones(4), np.zeros(4)]),
                    #   np.hstack([np.zeros(3), np.ones(5)]),
                    #   [1, 0, 0, 1, 1, 1, 0, 0]
                    ]

    for ax, values, name in zip(axes, binary_patterns, titles):
        plot_lbp_model(ax, values)
        ax.set_title(name)
    plt.show()

from skimage.transform import rotate
from skimage.feature import local_binary_pattern
from skimage import data
from skimage.color import label2rgb


def overlay_labels(image, lbp, labels):
    mask = np.logical_or.reduce([lbp == each for each in labels])
    return label2rgb(mask, image=image, bg_label=0, alpha=0.5)


def highlight_bars(bars, indexes):
    for i in indexes:
        bars[i].set_facecolor('r')


def hist(ax, lbp):
    n_bins = int(lbp.max() + 1)
    return ax.hist(lbp.ravel(), density=True, bins=n_bins, range=(0, n_bins),
                   facecolor='0.5')

def lbp_test():
    # settings for LBP
    radius = 1
    n_points = 8 * radius
    #n_points = 64
    #image = data.brick()
    image = cv2.imread(r'D:\BoyangDeng\BlueberryClassification\datasets\BlueberryScansforDestructiveTesting06142022\BadBlueberryScans\im_data_local\155\211_673.29.png')
    if len(image.shape) == 3:
        image = image[:, :, 0]
    lbp = local_binary_pattern(image, n_points, radius, METHOD)

    # plot histograms of LBP of textures
    fig, (ax_img, ax_hist) = plt.subplots(nrows=2, ncols=3, figsize=(9, 6))
    plt.gray()

    titles = ('edge', 'flat', 'corner')
    w = width = radius - 1
    edge_labels = range(n_points // 2 - w, n_points // 2 + w + 1)
    flat_labels = list(range(0, w + 1)) + list(range(n_points - w, n_points + 2))
    i_14 = n_points // 4            # 1/4th of the histogram
    i_34 = 3 * (n_points // 4)      # 3/4th of the histogram
    corner_labels = (list(range(i_14 - w, i_14 + w + 1)) +
                    list(range(i_34 - w, i_34 + w + 1)))

    label_sets = (edge_labels, flat_labels, corner_labels)

    for ax, labels in zip(ax_img, label_sets):
        ax.imshow(overlay_labels(image, lbp, labels))

    for ax, labels, name in zip(ax_hist, label_sets, titles):
        counts, _, bars = hist(ax, lbp)
        highlight_bars(bars, labels)
        ax.set_ylim(top=np.max(counts[:-1]))
        ax.set_xlim(right=n_points + 2)
        ax.set_title(name)

    ax_hist[0].set_ylabel('Percentage')
    for ax in ax_img:
        ax.axis('off')
    plt.show()
    
def main():
    lbp_test()

if __name__ =='__main__':
    main()
