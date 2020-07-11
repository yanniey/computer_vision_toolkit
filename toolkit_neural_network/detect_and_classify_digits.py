import cv2
import numpy as np

import neural_net_MNIST # import the neural net model we've built from MNIST data

OPENCV_MAJOR_VERSION = int(cv2.__version__.split('.')[0])

# helper function: determine if one bounding rectangle is entirely contained inside another bounding rectangle
# this allows us to only choose the outer bounding box for each digit 
def inside(r1, r2):
    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2
    return (x1 > x2) and (y1 > y2) and (x1+w1 < x2+w2) and \
            (y1+h1 < y2+h2)

# helper function: convert a tight-fitting bounding rectangle into a square with padding around the digit
# This ensures that skinny digits (such as 1) and fat digits (such s 0) are not streched differently
def wrap_digit(rect, img_w, img_h):

    x, y, w, h = rect

    x_center = x + w//2
    y_center = y + h//2
    # modify the rectangle's smaller dimension (be it width or height) so that it equals the larger dimension, and we modify the rectangle's x or y position so that the center remains unchanged
    if (h > w):
        w = h
        x = x_center - (w//2)
    else:
        h = w
        y = y_center - (h//2)

    # add 5-pixel padding on all sides
    padding = 5
    x -= padding
    y -= padding
    w += 2 * padding
    h += 2 * padding

    # crop bounding boxes so that they lie entirely within the image. May result in non-square rectangles
    if x < 0:
        x = 0
    elif x > img_w:
        x = img_w

    if y < 0:
        y = 0
    elif y > img_h:
        y = img_h

    if x+w > img_w:
        w = img_w - x

    if y+h > img_h:
        h = img_h - y

    # return the modified rectangle's coordinates
    return x, y, w, h


ann, test_data = neural_net_MNIST.train(
    neural_net_MNIST.create_ann(60), 50000, 10)

img_path = "../dataset/digit_images/digits_0.jpg"
img = cv2.imread(img_path, cv2.IMREAD_COLOR)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.GaussianBlur(gray, (7, 7), 0, gray) # blur it to remove noise and make the darkness of the ink more uniform


# apply a threshold and some morphology operations to ensure that the numbers stand out from the background and that the contours are relatively free of irregularities
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
erode_kernel = np.ones((2, 2), np.uint8)
thresh = cv2.erode(thresh, erode_kernel, thresh, iterations=2)

if OPENCV_MAJOR_VERSION >= 4:
    # OpenCV 4 or a later version is being used.
    contours, hier = cv2.findContours(thresh, cv2.RETR_TREE,
                                      cv2.CHAIN_APPROX_SIMPLE) # find contours
else:
    # OpenCV 3 or an earlier version is being used.
    # cv2.findContours has an extra return value.
    # The extra return value is the thresholded image, which is
    # unchanged, so we can ignore it.
    _, contours, hier = cv2.findContours(thresh, cv2.RETR_TREE,
                                         cv2.CHAIN_APPROX_SIMPLE)



#  iterate through the contours and find their bounding rectangles. 
#  We discard any rectangles that we deem too large or too small to be digits. 
#  We also discard any rectangles that are entirely contained in other rectangles. 
#  The remaining rectangles are appended to a list of good rectangles, which (we believe) contain individual digits
rectangles = []

img_h, img_w = img.shape[:2]
img_area = img_w * img_h
for c in contours:

    a = cv2.contourArea(c)
    if a >= 0.98 * img_area or a <= 0.0001 * img_area:
        continue

    r = cv2.boundingRect(c)
    is_inside = False
    for q in rectangles:
        if inside(r, q):
            is_inside = True
            break
    if not is_inside:
        rectangles.append(r)


# Now that we have a list of good rectangles, we can iterate through them, sanitize them using our wrap_digit function, and classify the image data inside them
for r in rectangles:
    x, y, w, h = wrap_digit(r, img_w, img_h)
    roi = thresh[y:y+h, x:x+w]
    digit_class = int(neural_net_MNIST.predict(ann, roi)[0])
    cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 2) # draw the sanitized bounding rectangle and the classification result
    cv2.putText(img, "%d" % digit_class, (x, y-5),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

cv2.imwrite("detected_and_classified_digits_thresh.png", thresh)
cv2.imwrite("detected_and_classified_digits.png", img)
cv2.imshow("thresh", thresh)
cv2.imshow("detected and classified digits", img)
cv2.waitKey()
