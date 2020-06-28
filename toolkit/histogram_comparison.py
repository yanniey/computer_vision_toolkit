import cv2
import numpy as np
from matplotlib import pyplot as plt

img1 = cv2.imread("../images/sky.jpeg")
img1_hist = cv2.calcHist([img1],[0], # focus on the blue channel only
                         None, # optional mask function
                         [256], # how many bins for the histogram x-axis
                         [0,256] # range of histogram
                        )

img2 = cv2.imread('../images/kitten.jpeg')
img2_hist = cv2.calcHist([img2],[0],None,[256],[0,256])

# calculate correlation with compareHist()
img1_2 = cv2.compareHist(img1_hist, img2_hist,0) # 0 is the correlation comparison method