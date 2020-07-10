# customised object detector with sliding window 
# Bag of Words + SVM + sliding window + non-max suppression

import cv2
import numpy as np
import os

from non_max_suppression import non_max_suppression_fast as nms

BOW_NUM_TRAINING_SAMPLES_PER_CLASS = 10
SVM_NUM_TRAINING_SAMPLES_PER_CLASS = 100

SVM_SCORE_THRESHOLD = 1.8 # threshold between a positive window and a negative window
NMS_OVERLAP_THRESHOLD = 0.15 # max acceptable proportion of overlap in the non-max suppression step. Adjustable




# load images
def get_pos_and_neg_paths(i):
    pos_path = '../../dataset/UIUCCarDetection/TrainImages/pos-%d.pgm' % (i+1)
    neg_path = '../../dataset/UIUCCarDetection/TrainImages/neg-%d.pgm' % (i+1)
    return pos_path, neg_path


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
   



# set 12 clusters for k-means
bow_kmeans_trainer = cv2.BOWKMeansTrainer(12)

# adjust SVM parameters
svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setC(50) # bigger C -> risk of false positives goes down, but risk of false negative increases
svm.train(np.array(training_data), cv2.ml.ROW_SAMPLE,
          np.array(training_labels))
svm.save('my_svm.xml') # save the detector to file
# svm.load('my_svm.xml') # load a saved SVM detector

# helper function: generate image pyramid
def pyramid(img, scale_factor=1.25, min_size=(200, 80),
            max_size=(600, 600)):
    h, w = img.shape
    min_w, min_h = min_size
    max_w, max_h = max_size
    while w >= min_w and h >= min_h:
        if w <= max_w and h <= max_h:
            yield img
        w /= scale_factor
        h /= scale_factor
        img = cv2.resize(img, (int(w), int(h)),
                         interpolation=cv2.INTER_AREA)

# helper function:  generate regions of interest based on sliding window 
# given an image, return the upper-left coordinates and the sub-image representing the next window. Successive windows are shifted by an arbitrarily sized step from left to right until we reach the end of a row, and from the top to bottom until we reach the end of the image.
def sliding_window(img, step=20, window_size=(100, 40)):
    img_h, img_w = img.shape
    window_w, window_h = window_size
    for y in range(0, img_w, step):
        for x in range(0, img_h, step):
            roi = img[y:y+window_h, x:x+window_w]
            roi_h, roi_w = roi.shape
            if roi_w == window_w and roi_h == window_h:
                yield (x, y, roi)

# for each test image, iterate over image pyramid, and for each pyramid level, we iterate over the sliding window positions
for test_img_path in ['../../dataset/UIUCCarDetection/TestImages/test-0.pgm',
                      '../../dataset/UIUCCarDetection/TestImages/test-1.pgm',
                      '../../images/people.jpeg',
                      '../../images/tree.jpeg']:
    img = cv2.imread(test_img_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pos_rects = []
    for resized in pyramid(gray_img):
        for x, y, roi in sliding_window(resized):
            descriptors = extract_bow_descriptors(roi) # For each window or region of interest (ROI), we extract BoW descriptors and classify them using the SVM
            if descriptors is None:
                continue
            prediction = svm.predict(descriptors)
            if prediction[1][0][0] == 1.0:
                raw_prediction = svm.predict(
                    descriptors, 
                    flags=cv2.ml.STAT_MODEL_RAW_OUTPUT) # must run predict method with this flag to get a confidence score
                score = -raw_prediction[1][0][0] # a lower value of the returned score represents high confidence, so we invert the score to make it more intuitive & suitable for non-max supression
                if score > SVM_SCORE_THRESHOLD: # If the classification produces a positive result that passes a certain confidence threshold, we add the rectangle's corner coordinates and confidence score to a list of positive detections
                    h, w = roi.shape
                    scale = gray_img.shape[0] / \
                        float(resized.shape[0])
                    pos_rects.append([int(x * scale), # rescale the coorindate back to the original images' scale
                                      int(y * scale),
                                      int((x+w) * scale),
                                      int((y+h) * scale),
                                      score]) # 
    

    # apply non-max suppression to get the highest-scoring rectangles among the ones that overlap due to sliding window
    pos_rects = nms(np.array(pos_rects), NMS_OVERLAP_THRESHOLD)

    # draw the rectangles and scores
    for x0, y0, x1, y1, score in pos_rects:
        cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)),
                      (0, 255, 255), 2)
        text = '%.2f' % score
        cv2.putText(img, text, (int(x0), int(y0) - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    cv2.imshow(test_img_path, img)
cv2.waitKey()
cv2.destroyAllWindows()