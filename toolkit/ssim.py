import numpy as np
import matplotlib.pyplot as plt

from skimage import io, data, color, img_as_float
from skimage.metrics import structural_similarity

sky = img_as_float(io.imread("../images/sky.jpeg"))

## add noise to image
noise = np.ones_like(sky) * 0.4 * (sky.max() - sky.min())
noise[np.random.random(size=sky.shape) > 0.7] *= -1
sky_noisy = sky + noise

## add absolute noise to image
sky_noisy_constant = sky + abs(noise)

## Function to calculate the mean square error (L2 distance) between images
def mse(x,y):
    return np.linalg.norm(x-y)


# MSE (L2 distance)
mse_noise = mse(sky,sky_noisy)

# SSI
ssim_noise = structural_similarity(sky, 
                                   sky_noisy,
                                   multichannel = True,
                                   data_range = sky_noisy.max() - sky_noisy.min())


# MSE (L2 distance)
mse_constant = mse(sky,sky_noisy_constant)

# SSI
ssim_constant = structural_similarity(sky, 
                                   sky_noisy_constant,
                                   multichannel = True,
                                   data_range = sky_noisy_constant.max() - sky_noisy_constant.min())


# plot the difference

fig, axes = plt.subplots(1,3,
                       figsize=(14,6),
                       sharex=True,sharey=True)

ax = axes.ravel()

label = 'MSE: {:.2f}, SSIM: {:.2f}'
ax[0].imshow(sky)
ax[1].set_xlabel("MSE:0, SSIM:1",fontsize=20)
ax[0].set_title('Original')

ax[1].imshow(sky_noisy)
ax[1].set_xlabel(label.format(mse_noise,ssim_noise),fontsize=20)
ax[1].set_title('Noisy')

ax[2].imshow(sky_noisy_constant)
ax[2].set_xlabel(label.format(mse_constant,ssim_constant),fontsize=20)
ax[2].set_title('Constant')