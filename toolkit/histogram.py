import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("../images/sky.jpeg")
cv2.imshow("sky", img)
cv2.waitKey()
cv2.destroyAllWindows()


img_hist = cv2.calcHist([img],[0],None,[256],[0,256])
plt.plot(img_hist)
plt.show()