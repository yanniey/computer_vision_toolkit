#  Each Haar-like feature describes the pattern of contrast among adjacent image regions. For example, edges, vertices, and thin lines each generate a kind of feature. Some features are distinctive in the sense that they typically occur in a certain class of object (such as a face) but not in other objects. These distinctive features can be organized into a hierarchy, called a cascade, in which the highest layers contain features of greatest distinctiveness, enabling a classifier to quickly reject subjects that lack these features.


# Haar cascades, as implemented in OpenCV, are not robust to changes in rotation or perspective. 


import cv2

face_cascade = cv2.CascadeClassifier(
    '../dataset/HaarCascades/haarcascade_frontalface_default.xml')
img = cv2.imread('../images/irish_passport.jpg')

# convert image to gray as CascadeClassifier expects grayscale images
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 
	scaleFactor = 1.08,  #scaleFactor should be >1. It determines the downscaling ratio of the image at each iteration of the face detection process
	minNeighbors = 5) # minimum number of overlapping detections that are required in order to retain a detection result. the bigger it is, the more confident we are about the detection

# visualise the area detected as face
for (x, y, w, h) in faces:
    img = cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
  
cv2.namedWindow('Face Detected!')
cv2.imshow('Face Detected!', img)
cv2.imwrite('./face_detected.jpg', img)
cv2.waitKey(0)