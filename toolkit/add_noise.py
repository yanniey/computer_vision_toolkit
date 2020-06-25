import numpy as np
import matplotlib.pyplot as plt

from skimage import io, data, color, img_as_float

sky = img_as_float(io.imread("images/sky.jpeg"))


noise = np.ones_like(sky) * 0.4 * (sky.max() - sky.min())
noise[np.random.random(size=sky.shape) > 0.7] *= -1


sky_noisy = sky + noise

plt.figure(figsize=(8,8))
plt.imshow(sky_noisy)