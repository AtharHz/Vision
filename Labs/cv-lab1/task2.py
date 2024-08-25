import numpy as np
from matplotlib import pyplot as plt
I = plt.imread('masoleh_gray.jpg')
# Iprim=np.flip(I,axis=0)
Iprim=I[::-1]
I=np.concatenate((I,Iprim))
plt.imshow(I,cmap='gray')
plt.show()