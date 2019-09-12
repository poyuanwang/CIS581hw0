import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from PIL import Image
import matplotlib.image as mpimg
import cv2

# -- Read an image --
# Attribution - Bikesgray.jpg By Davidwkennedy (http://en.wikipedia.org/wiki/File:Bikesgray.jpg) [CC BY-SA 3.0 (http://creativecommons.org/licenses/by-sa/3.0)], via Wikimedia Commons
img=Image.open("Bikesgray.jpg")
img1 = np.array(img)
# Write code here to read in the image named 'Bikesgray.jpg' into the variable img1

# -- Display original image --
plt.subplot(3, 1, 1)
plt.imshow(img1, cmap = 'gray')
plt.axis('off')
plt.title('bike orig image')


# -- X gradient - Sobel Operator --
f1 = np.asarray([[1, 0, -1], [2, 0, -2], [1, 0, -1]])

# -- Convolve image with kernel f1 -> This highlights the vertical edges in the image --
vertical_sobel = signal.convolve(img1, f1)
# Write code here to convolve img1 with f1

# -- Display the image --
plt.subplot(3, 1, 2)
plt.imshow(vertical_sobel, cmap = 'gray')
plt.axis('off')
plt.title('vertical edges image')

# Write code here to display the image 'vertical_sobel' (hint: use plt.imshow with a color may of gray)

# -- Y gradient - Sobel Operator --
f2 =  np.asarray([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])# Now if you want to highlight horizontal edges in the image, think about what the kernel should be. Store this kernel in the variable f2.

# -- Convolve image with kernel f2 -> This should highlight the horizontal edges in the image --
horz_sobel = signal.convolve(img1, f2)
# Write code here to convolve img1 with f2

# -- Display the image --
plt.subplot(3, 1, 3)
plt.imshow(horz_sobel, cmap = 'gray')
plt.axis('off')
plt.title('horizontal edges image')
# Write code here to display the image 'horz_sobel' (hint: use plt.imshow with a color may of gray)
plt.show()

