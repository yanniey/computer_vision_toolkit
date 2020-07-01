# segmenting foreground with GrabCut algo

# how GrabCut works:
# 1. A rectangle including the subject(s) of the picture is defined.
# 2. The area lying outside the rectangle is automatically defined as a background.
# 3. The data contained in the background is used as a reference to distinguish background areas from foreground areas within the user-defined rectangle.
# 4. A Gaussian Mixture Model (GMM) models the foreground and background, and labels undefined pixels as probable background and probable foreground.
# 5. Each pixel in the image is virtually connected to the surrounding pixels through virtual edges, and each edge is assigned a probability of being foreground or background, based on how similar it is in color to the pixels surrounding it.
# 6. Each pixel (or node as it is conceptualized in the algorithm) is connected to either a foreground or a background node. 
# 7. After the nodes have been connected to either terminal (the background or foreground, also called the source or sink, respectively), the edges between nodes belonging to different terminals are cut (hence the name, GrabCut). Thus, the image is segmented into two parts. 


import numpy as np
import cv2
from matplotlib import pyplot as plt

original = cv2.imread('../images/irish_passport_2.jpg')
img = original.copy()
mask = np.zeros(img.shape[:2], np.uint8) # create a mask that's the same shape as the image, but filled with zeros

# create zero-filled foreground and background models
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

# rectangle to initialise GrabCut
rect = (100, 1, 421, 378)

cv2.grabCut(img, 
			mask, 
			rect, 
			bgdModel, 
			fgdModel, 
			5,  # number of iterations
			cv2.GC_INIT_WITH_RECT)


# mask value interpretation:
# 0 = definitely background
# 1 = definitely foreground
# 2 = probably background
# 3 = probably foreground

# visualise GrabCut by turning the background pixels to black
mask2 = np.where((mask==2) | (mask==0), 0, 1).astype('uint8')
img = img*mask2[:,:,np.newaxis]



plt.subplot(121)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("grabcut")
plt.xticks([])
plt.yticks([])

plt.subplot(122)
plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
plt.title("original")
plt.xticks([])
plt.yticks([])

plt.show()