import cv2

grayImage = cv2.imread('../images/sky.jpeg', cv2.IMREAD_GRAYSCALE)
cv2.imwrite('sky.jpeg', grayImage)
