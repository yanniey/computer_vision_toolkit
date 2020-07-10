

import cv2
import numpy as np
 # Initialize the histogram.
 
x, y, w, h = track_window
roi = hsv_frame[y:y+h, x:x+w]
roi_hist = cv2.calcHist([roi], [0], None, [16], [0, 180])
self.roi_hist = cv2.normalize(roi_hist, roi_hist, 0, 255, # normalize hue histogram
                                cv2.NORM_MINMAX)