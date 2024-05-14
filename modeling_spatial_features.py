from IPython.display import clear_output
from IPython.display import HTML, display
import time
import sys
import math
import scipy.io
from scipy import signal
from matplotlib import pyplot as plt
import os
from skimage import feature
import numpy as np
import cv2
import cv2 as cv
# ((210, 39),

spatial_feature_means = None
spatial_feature_stds = None
# save_path_means = r'E:\Repo\Biang\Graphics\Blueberry\spatial_feature_means.npy'
# np.save(save_path_means, spatial_feature_means)
# spatial_feature_means = np.load(save_path_means, allow_pickle='TRUE')
# save_path_stds = r'E:\Repo\Biang\Graphics\Blueberry\spatial_feature_stds.npy'
# np.save(save_path_stds, spatial_feature_stds)
# spatial_feature_stds = np.load(save_path_means, allow_pickle='TRUE')

haralick_means = np.array([0.013, 49.8386, 0.9825, 1792.6416, 0.4318,
     138.8401, 7120.7279, 6.0121, 6.9556, 0.0012,
     2.1071, -0.5491, 0.9964, 0.005, 59.3764,
     0.9887, 2980.6533, 0.2964, 177.9784, 11863.2367,
     6.8831, 8.3007, 0.0008, 2.6903, -0.4976,
     0.9974, 0.0035, 101.0418, 0.9821, 3060.5144,
     0.2133, 184.4855, 12141.0158, 6.9845, 8.7375,
     0.0006, 3.1096, -0.4218, 0.9946])
haralick_stds = np.array([0.0054, 29.8301, 0.0119, 936.0835, 0.0553, 40.4582,
    3736.9489, 0.5091, 0.5883, 0.0002, 0.2356, 0.0403,
    0.0033, 0.0018, 24.5247, 0.0071, 1098.7116, 0.0311,
    38.6578, 4387.5463, 0.3928, 0.4442, 0.0001, 0.2111,
    0.026, 0.0024, 0.0013, 31.5947, 0.0084, 916.0063,
    0.0223, 33.4619, 3655.1298, 0.3555, 0.4331, 0.0001,
    0.2198, 0.026, 0.0034])
default_file_path = r'D:\BoyangDeng\Biang\Graphics\BlueberryClassification\texturefilters\ICAtextureFilters_7x7_8bit.mat'


def haralick_demo():
    # importing various libraries
    import mahotas
    import mahotas.demos
    import mahotas as mh
    import numpy as np
    from pylab import imshow, show

    # loading nuclear image
    nuclear = mahotas.demos.nuclear_image()
    # imshow(nuclear)
    # show(block=True)
    # filtering image
    nuclear = nuclear[:, :, 0]

    # adding gaussian filter
    nuclear = mahotas.gaussian_filter(nuclear, 4)

    # setting threshold
    threshed = (nuclear > nuclear.mean())
    imshow(threshed)
    show(block=True)
    # making is labeled image
    labeled, n = mahotas.label(threshed)

    # showing image
    print("Labelled Image")
    imshow(labeled)
    show(block=True)

    # getting haralick features
    h_feature = mahotas.features.haralick(labeled)

    # showing the feature
    print("Haralick Features")
    imshow(h_feature)
    show(block=True)


def ICA_demo():
    import plotly.express as px
    from sklearn.datasets import load_digits
    from sklearn.decomposition import FastICA
    X, _ = load_digits(return_X_y=True)
    print(X.shape, X.max(), X.dtype)
    px.imshow(X[0].reshape(8, 8))

    transformer = FastICA(n_components=7,
                          random_state=0,
                          whiten='unit-variance')
    X_transformed = transformer.fit_transform(X)
    print(X_transformed.shape)
    px.imshow(X_transformed[:10])


def create_gaborfilter():
    # This function is designed to produce a set of GaborFilters
    # an even distribution of theta values equally distributed amongst pi rad / 180 degree
    filters = []
    num_filters = 16
    ksize = 35  # The local area to evaluate
    sigma = 3.0  # Larger Values produce more edges
    lambd = 10.0
    gamma = 0.5
    psi = 0  # Offset value - lower generates cleaner results
    for theta in np.arange(0, np.pi, np.pi / num_filters):  # Theta is the orientation for edge detection
        kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_64F)
        kern /= 1.0 * kern.sum()  # Brightness normalization
        filters.append(kern)
    return filters


def apply_filter(img, filters):
    # This general function is designed to apply filters to our image

    # First create a numpy array the same size as our input image
    newimage = np.zeros_like(img)

    # Starting with a blank image, we loop through the images and apply our Gabor Filter
    # On each iteration, we take the highest value (super impose), until we have the max value across all filters
    # The final image is returned
    depth = -1  # remain depth same as original image

    for kern in filters:  # Loop through the kernels in our GaborFilter
        image_filter = cv2.filter2D(img, depth, kern)  # Apply filter to image

        # Using Numpy.maximum to compare our filter and cumulative image, taking the higher value (max)
        np.maximum(newimage, image_filter, newimage)
    return newimage


class LocalBinaryPatterns:
    def __init__(self, numPoints, radius):
        # store the number of points and radius
        self.numPoints = numPoints
        self.radius = radius

    def describe(self, image, eps=1e-7):
        # compute the Local Binary Pattern representation
        # of the image, and then use the LBP representation
        # to build the histogram of patterns
        # skimage.feature.local_binary_pattern(image, P, R, method='default')
        lbp = feature.local_binary_pattern(image, self.numPoints, self.radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, self.numPoints + 3), range=(0, self.numPoints + 2))
        # normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)
        # return the histogram of Local Binary Patterns
        return hist


def bsif(img, filterpath=default_file_path):
    f = scipy.io.loadmat(filterpath)
    texturefilters = f.get('ICAtextureFilters')

    img = img.astype("float")
    numScl = np.shape(texturefilters)[2]
    codeImg = np.ones(np.shape(img))

    # Make spatial coordinates for sliding window
    r = int(math.floor(np.shape(texturefilters)[0] / 2))

    # Wrap image (increase image size according to maximum filter radius by wrapping around)
    upimg = img[0:r, :]
    btimg = img[-r:, :]
    lfimg = img[:, 0:r]
    rtimg = img[:, -r:]
    cr11 = img[0:r, 0:r]
    cr12 = img[0:r, -r:]
    cr21 = img[-r:, 0:r]
    cr22 = img[-r:, -r:]
    imgWrap = np.vstack(
        (np.hstack((cr22, btimg, cr21)), np.hstack((rtimg, img, lfimg)), np.hstack((cr12, upimg, cr11))))

    # Loop over scales
    for i in range(numScl):
        tmp = texturefilters[:, :, numScl - i - 1]
        ci = signal.convolve2d(imgWrap, np.rot90(tmp, 2), mode='valid')
        t = np.multiply(np.double(ci > 0), 2 ** i)
        codeImg = codeImg + t

    hist_bsif = np.histogram(codeImg.ravel(), bins=np.arange(1, (2**numScl)+2))
    hist_bsif = hist_bsif[0]
    # normalize the histogram
    hist_bsif = hist_bsif/(hist_bsif.sum() + 1e-7)
    return codeImg, hist_bsif
