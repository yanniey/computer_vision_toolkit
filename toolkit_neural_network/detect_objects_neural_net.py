# Capture frames from a webcam in real-time and use a DNN to detect and classify 20 kinds of objects that may be in any given frame
# MobileNet- SSD (Single Shot Detector), whose output includes a subarray of detected objects, each with its own confidence score, rectangle coordinates, and class ID


import cv2
import numpy as np


model = cv2.dnn.readNetFromCaffe( # load the Caffe model
    '../dataset/mobileNet_SSD_data/MobileNetSSD_deploy.prototxt',
    '../dataset/mobileNet_SSD_data/MobileNetSSD_deploy.caffemodel')

# preprocessing 
blob_height = 300 # input image height
color_scale = 1.0/127.5 # convert data in the range [0,255] to range [-1.0,1.0]
average_color = (127.5, 127.5, 127.5)
confidence_threshold = 0.5
labels = ['airplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
          'car', 'cat', 'chair', 'cow', 'dining table', 'dog',
          'horse', 'motorbike', 'person', 'potted plant', 'sheep',
          'sofa', 'train', 'TV or monitor']


# For each frame, we begin by calculating the aspect ratio. This NN expects the input to be based on an image that is 300 pixels high
# However, the width can vary in order to match the original aspect ratio. The following code shows how we capture a frame and calculate the appropriate input size
cap = cv2.VideoCapture(0)

success, frame = cap.read()
while success:

    h, w = frame.shape[:2]
    aspect_ratio = w/h

    # Detect objects in the frame.
    blob_width = int(blob_height * aspect_ratio)
    blob_size = (blob_width, blob_height)
    # Perform preprocessing with cv2.dnn.blobFromImage
    blob = cv2.dnn.blobFromImage(
        frame, scalefactor=color_scale, size=blob_size,
        mean=average_color)

    # feed the preprocessed iage to the neural net and get the model's output
    model.setInput(blob)
    results = model.forward()

    # Iterate over the detected objects. The results from Single Shot Detector returns an array of detected objects, each with its own confidence score, rectangle coordinates, and class ID
    for object in results[0, 0]:
        confidence = object[2]
        if confidence > confidence_threshold:

            # Get the object's coordinates.
            x0, y0, x1, y1 = (object[3:7] * [w, h, w, h]).astype(int)

            # Get the classification result.
            id = int(object[1])
            label = labels[id - 1]

            # Draw a blue rectangle around the object.
            cv2.rectangle(frame, (x0, y0), (x1, y1),
                          (255, 0, 0), 2)

            # Draw the classification result and confidence.
            text = '%s (%.1f%%)' % (label, confidence * 100.0)
            cv2.putText(frame, text, (x0, y0 - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('Objects', frame)

    k = cv2.waitKey(1)
    if k == 27:  # Escape
        break

    success, frame = cap.read()
