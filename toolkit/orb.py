# orb with brute force matching & KNN, and apply the ratio test
import cv2
from matplotlib import pyplot as plt

# Load the images.
img0 = cv2.imread('../images/icon.png',
                  cv2.IMREAD_GRAYSCALE)
img1 = cv2.imread('../images/irish_passport_2.jpg',
                  cv2.IMREAD_GRAYSCALE)

# Perform ORB feature detection and description.
orb = cv2.ORB_create()
kp0, des0 = orb.detectAndCompute(img0, None)
kp1, des1 = orb.detectAndCompute(img1, None)

# Perform brute-force KNN matching.
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
pairs_of_matches = bf.knnMatch(des0, des1, k=2) # returns a list of lists: each inner list contains at least 1 match and <= k matches, sorted from best to worst match

# Sort the pairs of matches by distance (i.e. quality of match).
pairs_of_matches = sorted(pairs_of_matches, key=lambda x:x[0].distance) # sorts the outer list based on the distance score of the best matches


# Apply the ratio test.
matches = [x[0] for x in pairs_of_matches
           if len(x) > 1 and x[0].distance < 0.8 * x[1].distance] # set the threshold at 0.8 times the distance score of the second-best match. This returns a list of best matches that pass the test.


# Draw the best 25 matches.
img_matches = cv2.drawMatches(
    img0, kp0, img1, kp1, matches[:25], img1,
    flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

# Show the matches.
plt.imshow(img_matches)
plt.show()