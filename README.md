## A list of Jupyter notebooks and functions for Computer Vision tasks, using OpenCV, Pillow and Tensorflow

:sunglasses:

[Cheat Sheet: How to debug a Neural Network for Computer Vision problems in Tensorflow](https://gist.github.com/yanniey/50151e9717393dc79993868deb14b194)

### Jupyter Notebooks 
1. Masking & Image Manipulation
    * Block views & Pooling (Max/Mean/Median)
    * Contour Detection
    * Convex Hull
    * Edge detection using Roberts, Sobel and Canny edge detectors
2. Image Descriptors
    1. SIFT(Scale Invariant Feature Transform): for detecting blobs (a region of an image that greatly differs from its surrounding areas). 
    2. SURF: for detecting blobs. Improved from SIFT and uses Fast Hessian algo
    3. DAISY descriptors
    4. HOG descriptor (Histogram of Oriented Gradients) with non-max suppression NMS (`hog_detect_people.py`)
    5. Harris: for detecting corners (`harrisCornerDetection.py`)
    6. FAST (Features from Accelerated Segment Test): for detecting corners
    7. BRIEF: for detecting blobs
    8. ORB (Oriented FAST and Rotated BRIEF): for detecting a combination of corners and blobs, uses both FAST and BRIEF (`orb_knn.py`)
3. Algo for matching features 
    1. Brute-force matching (`cv2.BFMatcher` class, using KNN and ratio test) (`orb_knn.py`)
    2. FLANN-based (Fast Library for Approximate Nearest Neighbors) matching (`flann.py` & `flann_homography.py`)
4. Denoising Filters
    1. Total variation filter: based on the principal that signals with noise have high total variation
    2. Bilateral filter: good at preserving edges
    3. Wavelet denoising filter: good at preserving image quality
5. Morphological reconstruction
    1. Erosion (to find holes in image)
    2. Dilation (to find peaks in image)
6. Segmentation and Transformation 
    1. Global (e.g. otsu thresholding) vs local thresholding (e.g. cv.adaptiveThreshold): Thresholding: convert grayscale images to binary, or generally to segment objects from the background
    2. RAG (Region Adjacency Graph):  Used to segment areas of interest from the image. Each region in image is represented as a graph node in RAG, and weight of edge = difference between average colors of pixels in each region
    3. Watershed algos (Classic vs. Compact): Treats a **grayscale** image as a topographical map and finds lines between pixels of equal brightness. These lines are then used to segment the image into regions
    4. Transformation algorithms: warp, swirl from skimage
    5. Structural similarity index & MSE: measure how two images are different from each other
7. Dimension Reduction
    1. Dictionary Learning
    2. Convolution kernels
    3. Autoencoders

### Toolkit - Image
1. High Pass Filter(HPF) and Low Pass Filter (LPF) (`hpf.py`)
2. Canny edge detection(`canny.py`)
3. Find contours (`contours.py`)
4. Try alls threshold methods, e.g. itsu, isodata, mean, min (`try_all_threshold.py`)
5. RAG thresholding(`rag_thresholding.py`)
6. Segmentation with Watershed algos (`watershed_classic.py` and `watershed_compact.py`)
7. Rotate, scale and translate the image (`warp.py`)
8. Add noise to image(`add_noise.py`)
9. Find similarity between images(MSE, Structural Similarity Index)(`ssim.py`)
10. Histogram comparison (`histogram_comparison.py` using the `compareHist` function from opencv)
11. Detecting lines with `HoughLines` and `HoughLinesP`, circles with `HoughCircles` (`lineDetection.py`,`circleDetection.py`). Detecting other shapes can be done via combining `cv2.findContours` and `cv2.approxPolyDP`
12. Foreground segmentation with `GrabCut` 
13. Haar face detection `haarFaceDetection.py`
14. Face recognition: `generateImages.py` and `faceRecognition.py` (Eigenfaces, Fisherfaces,Local Binary Patterns Histograms)
15. Homography, i.e. find images that contain a specific icon (`icon_matcher` folder)
16. Non-max supression, used for detection with sliding windows where one object may get detected multiple times `non_max_suppression.py`
17. Customised object detector with SIFT, Bag of Words(BoW), SVM, sliding window and non-max suppression `detector_car_svm.py` and `detector_car_bow_sliding_window.py`
17. Save and load an SVM detector with `svm.save()` and `svm.load()`

### Toolkit - Video
1. Object tracking techniques:
    1. Background subtraction
        * Basic motion detection using background subtraction `basic_motion_detection.py`
        * MOG background subtractor `mog.py`
        * KNN background subtractor  `knn.py`
        * GMG background subtractor `gmg.py`
    2. Histogram back-projection with MeanShift or CamShift `meanshift.py`, `camshift.py`
2. Kalman filters `kalman.py`, `kalman_pedestrian_tracking.py`

### Toolkit - Neural Network
1. Simple neural network `simple_neural_net.py`, `neural_net_multiple_features.py`
2. Recognizing handwritten MNIST digits with neural network `neural_net_MNIST.py`. Run `test_neural_net_MNIST.py` to see the neural net's accuracy
3. Use the model built from MNIST data on new data `detect_and_classify_digits.py`
4. Ways to improve neural net performance:
    1. Experiment with the size of your training dataset, the number of hidden nodes, and the number of epochs until you find a peak level of accuracy
    2. Modify `neural_net_MNIST.create_ann` function so that it supports more than one hidden layer
    3. Try different activation functions. We have used `cv2.ml.ANN_MLP_SIGMOID_SYM`, but it isn't the only option; the others include `cv2.ml.ANN_MLP_IDENTITY`, `cv2.ml.ANN_MLP_GAUSSIAN`, `cv2.ml.ANN_MLP_RELU`, and `cv2.ml.ANN_MLP_LEAKYRELU`
    4. Try different training methods. We have used `cv2.ml.ANN_MLP_BACKPROP`. The other options include `cv2.ml.ANN_MLP_RPROP` and `cv2.ml.ANN_MLP_ANNEAL`
5. Save and load neural network models `save_and_load_neural_net.py`
6. Load a deep learning model for tensorflow `load_tf_model.py`
7. Detect and classify objects with 3rd party neural net: mobileNet + Single Shot Detector `detect_objects_neural_net.py`
8. Detect and classify faces with 3rd party neural nets: `detect_faces_neural_net.py`
    * Face detection using the Caffe model `res10_300x300_ssd_iter_140000` 
    * Age and gender detection using the Caffe model `age_net` and `gender_net`

### Toolkit - Imitate Film Filters
1. Emulate the following 4 types of films using curves
    * Kodak Portra, a family of films that is optimized for portraits and weddings `class BGRPortraCurveFilter` in `filters.py`
    * Fuji Provia, a family of general-purpose films `class BGRProviaCurveFilter` in `filters.py`
    * Fuji Velvia, a family of films that is optimized for landscapes `class BGRVelviaCurveFilter` in `filters.py`
    * Cross-processing, a nonstandard film processing technique, sometimes used to produce a grungy look in fashion and band photography


### Workflow
1. Edge detection (e.g. Sobel, Canny). May need to convert to grayscale first 
2. Segment detection (e.g. RAG, watershed, GrabCut)
3. Transformation(rotation, scale, crop,distanceTransform)
    * Apply Gaussian blur to remove noise and make the darkness of image more uniform
    * Apply threshold  to make image stand out from the background, and erosion to make contours free of irregularities
4. Feature extraction
5. Feature matching
    * Brute Force
    * FLANN-based with KNN and ratio test

### Facial Detection and Recognition 
1. Haar cascade classifiers
2. Facial recognition: Eigenfaces, Fisherfaces, Local Binary Pattern Histograms (LBPHs)





