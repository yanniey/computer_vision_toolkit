# Save a neural net model
ann = cv2.ml.ANN_MLP_create()
data = cv2.ml.TrainData_create(
    training_samples, layout, training_responses)
ann.train(data)
ann.save('my_ann.xml')

# Load a neural net model
ann = cv2.ml.ANN_MLP_create()
ann.load('my_ann.xml')