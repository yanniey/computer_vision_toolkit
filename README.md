## A list of notebooks and helper functions for Computer Vision tasks

:sunglasses:

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



### Workflow
1. Edge detection (e.g. Sobel, Canny). May need to convert to grayscale first 
2. Segment detection (e.g. RAG, watershed, GrabCut)
3. Transformation(rotation, scale, crop,distanceTransform)
4. Feature extraction

### Facial Detection and Recognition 
1. Haar cascade classifiers
2. Facial recognition: Eigenfaces, Fisherfaces, Local Binary Pattern Histograms (LBPHs)
