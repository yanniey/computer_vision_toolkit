# create a customised object dector with bag of words and SVM
# SIFT descriptor + FLANN based matching
# Data set: UIUC Image Database for Car Detection https://cogcomp.seas.upenn.edu/Data/Car/
import cv2
import numpy as np
import os

# check if folder exists
if not os.path.isdir('../dataset/UIUCCarDetection'):
    print(
        'CarData folder not found. Please download and unzip '
        'http://l2r.cs.uiuc.edu/~cogcomp/Data/Car/CarData.tar.gz '
        'into the same folder as this script.')
    exit(1)


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

# load images
def get_pos_and_neg_paths(i):
    pos_path = '../dataset/UIUCCarDetection/TrainImages/pos-%d.pgm' % (i+1)
    neg_path = '../dataset/UIUCCarDetection/TrainImages/neg-%d.pgm' % (i+1)
    return pos_path, neg_path