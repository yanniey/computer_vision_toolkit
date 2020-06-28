import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, io, exposure, color
from skimage.util import crop

img1 = io.imread("images/irish_passport.jpg")
print("original image size is: ", img1.shape)
cropped_image = crop(img1, ((200,200),(100,100),(0,0)),copy=False)
print("cropped image size is: ", cropped_image.shape)

plt.imshow(cropped_image)