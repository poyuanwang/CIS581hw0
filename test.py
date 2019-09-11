import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from PIL import Image
import matplotlib.image as mpimg
import cv2

# -- Read an image --
# Attribution - Bikesgray.jpg By Davidwkennedy (http://en.wikipedia.org/wiki/File:Bikesgray.jpg) [CC BY-SA 3.0 (http://creativecommons.org/licenses/by-sa/3.0)], via Wikimedia Commons
image = Image.open("Bikesgray.jpg")
# -- Display original image --
image.show()
# -- X gradient - Sobel Operator --
f1 = np.asarray([[1, 0, -1], [2, 0, -2], [1, 0, -1]])

# -- Convolve image with kernel f1 -> This highlights the vertical edges in the image --

image = cv2.imread('Bikesgray.jpg', cv2.IMREAD_GRAYSCALE).astype(float) / 255.0
vertical_sobel = cv2.filter2D(src=image, kernel=f1, ddepth=-1) # Write code here to convolve img1 with f1


# -- Display the image --
# Write code here to display the image 'vertical_sobel' 
cv2.imshow('vertical edges', vertical_sobel)

# -- Y gradient - Sobel Operator --
f2 = np.array([[1, 2,1], [0, 0, 0], [-1, -2, -1]])  # Now if you want to highlight horizontal edges in the image, think about what the kernel should be. Store this kernel in the variable f2.

# -- Convolve image with kernel f2 -> This should highlight the horizontal edges in the image --
horz_sobel =cv2.filter2D(src=image, kernel=f2, ddepth=-1)  # Write code here to convolve img1 with f2

# -- Display the image --
cv2.imshow('horizontal edges', horz_sobel)

cv2.waitKey(0)
cv2.destroyAllWindows()