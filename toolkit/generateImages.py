# this script generates images from video camera. Stops when user presses any key
import cv2
import os


output_folder = '../dataset/at/aggg'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# load face and eye cascade
face_cascade = cv2.CascadeClassifier(
    '../dataset/HaarCascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(
    '../dataset/HaarCascades/haarcascade_eye.xml')

camera = cv2.VideoCapture(0)
count = 0
while (cv2.waitKey(1) == -1):
    success, frame = camera.read()
    if success:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.3, minNeighbors=5, minSize=(120, 120))
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            face_img = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
            face_filename = '%s/%d.pgm' % (output_folder, count)
            cv2.imwrite(face_filename, face_img)
            count += 1
        cv2.imshow('Capturing Faces...', frame)