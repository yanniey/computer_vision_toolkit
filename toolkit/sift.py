# SIFT = Sclae Invarient Feature Transform


# DAISY
# similar to SIFT, but faster and works with lower dimensionality feature vectors

# HOG
# feature descriptor used for object detection


# SIFT implementation is in this version of the opencv library 
! pip install opencv-contrib-python==3.4.2.16
import cv2
import skimage
import matplotlib.pyplot as plt
from skimage.transform import resize

# first image: detect and compute SIFT in two separate steps
img = cv2.imread("../images/irish_passport.jpg")
gray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
sift = cv2.xfeatures2d.SIFT_create()

kp1 = sift.detect(gray1, None) # None = no mask
kp1,des1=sift.compute(gray1,kp1)

# second image: detect and compute SIFT in one step
img2 = cv2.imread('images/irish_passport_2.jpg')
gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
kp2,des2=sift.detectAndCompute(gray2, None) # mask = None


## match features across different images
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
matches = bf.match(des1, des2)


# visualise the first 50 matches

n_matches = 50

match_img = cv2.drawMatches(
    gray1, kp1,
    gray2, kp2,
    matches[:n_matches],gray2.copy(),flags=0
)

plt.figure(figsize=(20,10))
plt.imshow(match_img)