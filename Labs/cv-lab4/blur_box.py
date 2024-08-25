import numpy as np
import cv2
from cv2.typing import Point

I = cv2.imread('isfahan.jpg').astype(np.float64) / 255;

# display the original image
cv2.imshow('original',I)
cv2.waitKey()

# creating a box filter
m = 7 # choose filter size

# create an m by m box filter
F = np.ones((m,m), np.float64)/(m*m)
print(F)

# Now, filter the image
# J = cv2.filter2D(I,-1, F)
# J = cv2.blur(I,(m,m))
J = cv2.boxFilter(I,-1,(m,m))
cv2.imshow('blurred', J)
cv2.waitKey()

cv2.destroyAllWindows()
