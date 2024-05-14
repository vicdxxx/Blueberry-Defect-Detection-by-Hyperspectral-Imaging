from spectral import *
from  pathlib import Path
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

data_dir = r'E:\PHD\Blueberry\Blueberry\Blueberry Scans for Destructive Testing 06132022'
data_dir = Path(data_dir) 
good_dir = data_dir / 'Good Blueberry Scans'

names = [ 'Good Blueberry 1-42', 'Good Blueberry 43-84']

ref_names= ['WhiteReference']
path = good_dir / f'{names[0]}.bil'.__str__()
path_head = good_dir / f'{names[0]}.bil.hdr'.__str__()

path1 = good_dir / f'{names[1]}.bil'.__str__()
path_head1 = good_dir / f'{names[1]}.bil.hdr'.__str__()

ref_path = data_dir / f'{ref_names[0]}.bil'.__str__()
ref_path_head = data_dir / f'{ref_names[0]}.bil.hdr'.__str__()

from spectral.io import envi
data_good = envi.open(path_head, path)
data_good1 = envi.open(path_head1, path1)
ref_data = envi.open(ref_path_head, ref_path)

data = data_good
wavelengths = data.metadata['wavelength']
wavelengths = [round(float(x)) for x in wavelengths]
data_num = len(wavelengths)
title_window = 'HSI corrected'

import plotly.express as px
import numpy as np

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx

def get_calibrated_image(target_wavelength , show=0):
    band, idx = find_nearest(wavelengths, target_wavelength)
    print(band, idx)
    idx=int(idx)
    im = data[:, :, idx]
    ref = ref_data[:, :, idx]
    im = im / ref
    if show:
        im = (im-im.min()) / (im.max()-im.min())
        im_np = np.array(im*255, dtype=np.uint8).squeeze()
        fig = px.imshow(im_np)
        fig.show()
    return im

def im_norm_255(im):
    im = (im-im.min()) / (im.max()-im.min())
    im_np = np.array(im*255, dtype=np.uint8).squeeze()
    return im_np

def show_im(im, norm=True):
    if norm:
        im_np = im_norm_255(im)
        fig = px.imshow(im_np)
        fig.show()
    else:
        fig = plt.imshow(im)
        plt.show()

im1 = get_calibrated_image(428)
im2 = get_calibrated_image(900)
diff_im = im2-im1
diff_im = cv.normalize(diff_im, diff_im, 0,255, cv.NORM_MINMAX)
diff_im = np.round(diff_im).astype(np.uint8)
equ = cv.equalizeHist(diff_im)

ret, im_t = cv.threshold(equ, 190, 255, cv.THRESH_BINARY)

# for convinenent use larger filter
#kernel=cv.getStructuringElement(cv.MORPH_RECT,(20,20))
kernel=cv.getStructuringElement(cv.MORPH_RECT,(5,5))
iterations = 1
eroded=cv.erode(im_t,kernel,iterations=iterations)
kernel=cv.getStructuringElement(cv.MORPH_RECT,(3,3))
#iterations = 10
iterations = 2
im_t_dilated=cv.dilate(eroded,kernel, iterations=iterations)

offsets = {'crop_x_start' : 170, 'crop_y_start': 50}

crop_y_start = offsets['crop_y_start']
crop_x_start = offsets['crop_x_start']

im_t = im_t_dilated[crop_y_start:, crop_x_start:]
im_base = equ[crop_y_start:, crop_x_start:]

#im_copy = im_base.copy()
im_copy = np.zeros_like(im_base)
contours, hierarchy = cv.findContours(image=im_t, mode=cv.RETR_EXTERNAL,
                                    method=cv.CHAIN_APPROX_SIMPLE)

im = im_t
im_copy = im_base.copy()
font=cv.FONT_HERSHEY_DUPLEX
center_pts = []
center_refs = []
for i_contour, contour in enumerate(contours):
    x = int(contour[:, :, 0].mean())
    y = int(contour[:, :, 1].mean())
    center_refs.append(x+10*y)
    center_pts.append((x, y))
idx_sorted = np.argsort(center_refs)
for i_idx_sorted, idx in enumerate(idx_sorted):
    x, y = center_pts[idx]
    cv.putText(im_copy, str(i_idx_sorted+1) , (x-5,y+5), font, 1, 0, 2)
print(len(contours))

import cv2 as cv
from tqdm import tqdm

hyper_dict = {}
for idx in tqdm(range(data_num), ncols=80):
    if idx < 350:
        continue
    im = data[: , :, idx]
    ref = ref_data[: , :, idx]
    im_corrected = im/ref
    im_corrected =  im_corrected[crop_y_start:, crop_x_start:]

    #show_im(im_corrected)

    for i_idx_sorted, idx in enumerate(idx_sorted):
        if i_idx_sorted!=26:
            continue
        contour = contours[idx]
        mask = np.zeros_like(im_corrected)
        mask = mask.astype(np.uint8)
        cv.drawContours(mask, [contour], -1, 255,-1)
        #cv.namedWindow(title_window, 0)
        #cv.imshow(title_window, mask)
        #cv.waitKey(0)
        #show_im(mask)
        #im_corrected_masked = cv.bitwise_and(im_norm_255(im_corrected), im_norm_255(im_corrected), mask=mask)
        #show_im(im_corrected_masked)
        pts  = np.where(mask == 255)
        roi = im_corrected[pts[0], pts[1]]

        y0 = pts[0].min()
        y1 = pts[0].max()
        x0 = pts[1].min()
        x1 = pts[1].max()
        #im_corrected_masked = cv.bitwise_and(im_norm_255(im_corrected), im_norm_255(im_corrected), mask=mask)
        one_sample = im_corrected[y0:y1+1, x0:x1+1]
        print(one_sample.min(), one_sample.max())
        show_im(one_sample, norm=False)

        mean_spectral = round(np.mean(roi), 4)
        if i_idx_sorted not in hyper_dict:
            hyper_dict[i_idx_sorted] = []
        hyper_dict[i_idx_sorted].append(mean_spectral)
key = list(hyper_dict.keys())[0]
y = hyper_dict[key]
x = wavelengths
import plotly.graph_objects as go	# 引入plotly底层绘图库

fig = go.Figure()				
fig.add_trace(go.Scatter(	
    x=x,			
    y=y,
    #text=y,
    #textposition="top center",
    #mode="markers+lines+text",
))
fig.show()						# 展示图表