import cv2

img = cv2.imread('../images/irish_passport.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
dst = cv2.cornerHarris(gray, 
	2, 
	7, # block size of the Sobel operator. Determines how sensitive corner detection is. 
	    # Must be an odd value between 3 and 31. The higher the value, the fewer corner it returns
	0.04)
img[dst > 0.01 * dst.max()] = [0, 0, 255] # select pixels with scores that are at least 1% of the highest score, and color these pixels red in the original image
cv2.imshow('corners', img)
cv2.waitKey()