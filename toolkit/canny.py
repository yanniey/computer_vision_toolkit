import cv2
import numpy as np

img = cv2.imread("../images/sky.jpeg", 0)
cv2.imwrite("canny.jpeg", cv2.Canny(img, 200, 300))
cv2.imshow("canny", cv2.imread("canny.jpeg"))
cv2.waitKey()
cv2.destroyAllWindows()
