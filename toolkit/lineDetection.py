# line detection can be implemented with Opencv's HoughLines function or the HoughLinesP function. The former uses the standard Hough transform, while the latter uses the probabilistic Hough transform (hence the P in the name). The probabilistic version is so-called because it only analyzes a subset of the image's points and estimates the probability that these points all belong to the same line. This implementation is an optimized version of the standard Hough transform; it is less computationally intensive and executes faster. HoughLinesP is implemented so that it returns the two endpoints of each detected line segment, whereas HoughLines is implemented so that it returns a representation of each line as a single point and an angle, without information about endpoints.
# use Canny or other denoising/edge detection methods before applying Hough transform
import cv2
import numpy as np

img = cv2.imread('../images/irish_passport.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 120)
minLineLength = 20 # lines shorter than this will be discarded
maxLineGap = 5 #  maximum size of a gap in a line before the two segments start being considered as separate lines
lines = cv2.HoughLinesP(edges, 
	                    rho=1,  # rho = step size in pixels
	                    theta = np.pi/180.0,  # step size in radian, i.g. 1 degree
	                    20, # thredhold. lines with < 20 votes are discarded
                        minLineLength, maxLineGap)
for x1, y1, x2, y2 in lines[0]:
    cv2.line(img, (x1, y1), (x2, y2), (0,255,0),2)

cv2.imshow("edges", edges)
cv2.imshow("lines", img)
cv2.waitKey()
cv2.destroyAllWindows()