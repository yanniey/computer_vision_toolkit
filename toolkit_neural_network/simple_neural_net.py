import cv2
import numpy as np

ann = cv2.ml.ANN_MLP_create() # create an untrained neural net
ann.setLayerSizes(np.array([9, 15, 9], np.uint8)) # of nodes in the input/hidden/output layer. [9,15,13,9] = 2 hidden layers with 15 and 13 nodes, respectively
ann.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM, 0.6, 1.0)
ann.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP, 0.1, 0.1)
ann.setTermCriteria(
    (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 100, 1.0))


# training
training_samples = np.array(
    [[1.2, 1.3, 1.9, 2.2, 2.3, 2.9, 3.0, 3.2, 3.3]], np.float32) # training input
layout = cv2.ml.ROW_SAMPLE # specify whether the training data's format is one row per sample or one column per sample
training_responses = np.array(
    [[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], np.float32) # training labels
data = cv2.ml.TrainData_create(
    training_samples, layout, training_responses)
ann.train(data)


# testing
test_samples = np.array(
    [[1.4, 1.5, 1.2, 2.0, 2.5, 2.8, 3.0, 3.1, 3.8]], np.float32)
prediction = ann.predict(test_samples)
print(prediction)
