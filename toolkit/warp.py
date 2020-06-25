import math
import matplotlib.pyplot as plt

from skimage import io, data, transform
from skimage.transform import swirl, warp

tree = io.imread("../images/tree.jpeg")

tform = transform.SimilarityTransform(scale = 1.5, rotation=math.pi/4,
                                     translation=(tree.shape[0]/2,-100))
rotated = warp(tree,tform) # use warp to specify the image & the transformation you want to apply
back_rotated = warp(tree,tform.inverse)

plt.figure(figsize=(8,8))
plt.imshow(rotated)