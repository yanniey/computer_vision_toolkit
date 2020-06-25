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


w0 = watershed(kitten_edge,seeds)

water_classic = color.label2rgb(w0,kitten,alpha=0.4,kind="overlay")

plt.figure(figsize=(8,8))
plt.imshow(water_classic)