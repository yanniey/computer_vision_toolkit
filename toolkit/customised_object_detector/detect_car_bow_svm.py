# create a customised object dector with bag of words and SVM. SVM is trained on BoW descriptors
# SIFT descriptor + FLANN based matching
# Data set: UIUC Image Database for Car Detection https://cogcomp.seas.upenn.edu/Data/Car/
import cv2
import numpy as np
import os


# load images
def get_pos_and_neg_paths(i):
    pos_path = '../../dataset/UIUCCarDetection/TrainImages/pos-%d.pgm' % (i+1)
    neg_path = '../../dataset/UIUCCarDetection/TrainImages/neg-%d.pgm' % (i+1)
    return pos_path, neg_path

BOW_NUM_TRAINING_SAMPLES_PER_CLASS = 10 # num of images 
SVM_NUM_TRAINING_SAMPLES_PER_CLASS = 100 # num of Bag of Word descriptors

# SIFT descriptor + FLANN based matching
sift = cv2.xfeatures2d.SIFT_create()

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = {}
flann = cv2.FlannBasedMatcher(index_params, search_params)

# train Bag of Word vocabulary, and covert lower level descriptor (SIFT) into Bag of Word descriptors
bow_kmeans_trainer = cv2.BOWKMeansTrainer(40)
bow_extractor = cv2.BOWImgDescriptorExtractor(sift, flann)

# helper function to load an image, extract SIFT descriptors, and add the SIFT descriptor to BoW vocabulary trainer
def add_sample(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    keypoints, descriptors = sift.detectAndCompute(img, None)
    if descriptors is not None:
        bow_kmeans_trainer.add(descriptors) # if no features are detected, then the keypoints and descriptors will be None

# add a few samples. car = positive class, not class = negative class
for i in range(BOW_NUM_TRAINING_SAMPLES_PER_CLASS):
    pos_path, neg_path = get_pos_and_neg_paths(i)
    add_sample(pos_path)
    add_sample(neg_path)

# k-means clustering on the BoW vocabulary
voc = bow_kmeans_trainer.cluster()
bow_extractor.setVocabulary(voc)

# helper function, returns an image BoW descriptor extractor 
def extract_bow_descriptors(img):
    features = sift.detect(img)
    return bow_extractor.compute(img, features)

# create a training set that are BoW descriptors
training_data = []
training_labels = []
for i in range(SVM_NUM_TRAINING_SAMPLES_PER_CLASS):
    pos_path, neg_path = get_pos_and_neg_paths(i)
    pos_img = cv2.imread(pos_path, cv2.IMREAD_GRAYSCALE)
    pos_descriptors = extract_bow_descriptors(pos_img)
    if pos_descriptors is not None:
        training_data.extend(pos_descriptors)
        training_labels.append(1) # label positive (car) samples as 1
    neg_img = cv2.imread(neg_path, cv2.IMREAD_GRAYSCALE)
    neg_descriptors = extract_bow_descriptors(neg_img)
    if neg_descriptors is not None:
        training_data.extend(neg_descriptors)
        training_labels.append(-1) # label negative (not car) samples as -1


# create an SVM and train it with BoW descriptor data
svm = cv2.ml.SVM_create()
svm.train(np.array(training_data), cv2.ml.ROW_SAMPLE,
          np.array(training_labels)) # must convert training data and labels from lists to numpy arrays before training


# run a few examples
for test_img_path in ['../../dataset/UIUCCarDetection/TestImages/test-0.pgm',
                      '../../dataset/UIUCCarDetection/TestImages/test-1.pgm',
                      '../../images/people.jpeg',
                      '../../images/tree.jpeg']:
    img = cv2.imread(test_img_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    descriptors = extract_bow_descriptors(gray_img)
    prediction = svm.predict(descriptors)
    if prediction[1][0][0] == 1.0:
        text = 'car'
        color = (0, 255, 0)
    else:
        text = 'not car'
        color = (0, 0, 255)
    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                color, 2, cv2.LINE_AA)
    cv2.imshow(test_img_path, img)
cv2.waitKey()
cv2.destroyAllWindows()