## A list of notebooks and helper functions for Computer Vision tasks

:sunglasses:

### Jupyter Notebooks 
1. Masking & Image Manipulation
    * Block views & Pooling (Max/Mean/Median)
    * Contour Detection
    * Convex Hull
    * Edge detection using Roberts, Sobel and Canny edge detectors
2. Image Descriptors
    1. SIFT(Scale Invariant Feature Transform)
    2. DAISY descriptors
    3. HOG descriptor (Histogram of Oriented Gradients)
3. Corner Detection
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

### Toolkit
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


### Workflow
1. Edge detection (e.g. Sobel, Canny). May need to convert to grayscale first
2. Segment detection (e.g. RAG, watershed, GrabCut)
3. Transformation(rotation, scale, crop,distanceTransform)
4. Feature extraction

### Facial Detection and Recognition 
1. Haar cascade classifiers
2. Facial recognition: Eigenfaces, Fisherfaces, Local Binary Pattern Histograms (LBPHs)
