import numpy as np
import matplotlib.pyplot as plt

from skimage import io, data, util,filters, color
from skimage.morphology import watershed

kitten = color.rgb2gray(io.imread("../images/kitten.jpeg"))

kitten_edge = filters.sobel(kitten) # use edge detection algo before watershed

grid = util.regular_grid(kitten.shape,n_points= 300) # find 300 points evenly spaced in the image 


# The seed matrix is the same shape as the original image, and it contains integers in the range [1, size of image]

seeds = np.zeros(kitten.shape,dtype=int) 
seeds[grid] = np.arange(seeds[grid].size).reshape(seeds[grid].shape)+1

w1 = watershed(kitten_edge, seeds, compactness = 0.91) ## compact watershed produces even region sizes
water_compact = color.label2rgb(w1,kitten,alpha=0.4,kind="overlay")

plt.figure(figsize=(8,8))
plt.imshow(water_compact)
