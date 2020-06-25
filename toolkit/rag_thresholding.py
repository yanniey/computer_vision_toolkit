from matplotlib import pyplot as plt
from skimage import data,io,segmentation,color
from skimage.future import graph


cats = io.imread("../images/three_cats.jpeg")
labels_1 = segmentation.slic(cats,
                             compactness=35, # balance color and space proximity. high value emphasies spatial closeness 
                             # and the segments are squarish
                             n_segments = 100)

seg_overlay = color.label2rgb(labels_1, cats, kind='overlay')

plt.figure(figsize=(8,8))
plt.imshow(seg_overlay)


g = graph.rag_mean_color(cats,labels_1)
labels_2 = graph.cut_threshold(labels_1, g, 
                               thresh=15 # combine regions seperates by a weight less than the threshold
                              )

seg_rag = color.label2rgb(labels_2, cats, kind='avg')

plt.figure(figsize=(8,8))
plt.imshow(seg_rag)
