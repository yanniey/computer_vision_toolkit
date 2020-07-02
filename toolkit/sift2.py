# cv2.SIFT = Difference of Gaussian feature detection + SIFT description
import cv2

img = cv2.imread('../images/varese.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create() # create a sift object and use the grayscale as input
keypoints, descriptors = sift.detectAndCompute(gray, None)

cv2.drawKeypoints(img, keypoints, img, (51, 163, 236),
                  cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS # draw a visualisation of the scale and orientation of each keypoint
                  )

cv2.imshow('sift_keypoints', img)
cv2.waitKey()