# uses cv2.HOGDescriptor to perform people detection. Uses SVM to produce confidence score

import cv2

def is_inside(i, o): # i = inner rectangle, o = outer rectangle
    ix, iy, iw, ih = i
    ox, oy, ow, oh = o
    return ix > ox and ix + iw < ox + ow and \
        iy > oy and iy + ih < oy + oh


# create an instance of HOG descriptor, set SVM to be the classifier
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


# load the image to detect people
img = cv2.imread('../images/people.jpeg')

found_rects, found_weights = hog.detectMultiScale(
    img, 
    winStride=(4, 4),  # stride of the sliding window. Smaller values --> more detections, but higher computational cost
    scale=1.02,  # scale factor in the image pyramid. Smaller values -> more detections, but higher computational cost. Must be > 1
    finalThreshold=1.9) # determines how stringent the detection criteria is. Smaller values -> more detections. Default = 2
    
    # this function returns 2 lists: list of bounding rectangles for detected objects, and a list of confidence scores for detected objects


# filter the detection results to remove nested rectangles (i.e. when one detected object is within another detected object)
found_rects_filtered = []
found_weights_filtered = []
for ri, r in enumerate(found_rects):
    for qi, q in enumerate(found_rects):
        if ri != qi and is_inside(r, q):
            break
    else:
        found_rects_filtered.append(r)
        found_weights_filtered.append(found_weights[ri])


# visualise the detected people
for ri, r in enumerate(found_rects_filtered):
    x, y, w, h = r
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
    text = '%.2f' % found_weights_filtered[ri]
    cv2.putText(img, text, (x, y - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

cv2.imshow('People Detected', img)
cv2.imwrite('../images/people_detected.jpg', img)
cv2.waitKey(0)