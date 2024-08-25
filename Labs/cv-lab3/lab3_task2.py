import cv2
import numpy as np
from matplotlib import pyplot as plt

fname = 'crayfish.jpg'
# fname = 'office.jpg'
# fname = 'map.jpg'
# fname = 'train.jpg'
# fname = 'branches.jpg'
# fname = 'terrain.jpg'

I = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)

f, axes = plt.subplots(2, 3)

axes[0,0].imshow(I, 'gray', vmin=0, vmax=255)
axes[0,0].axis('off')

hist=axes[1,0].hist(I.ravel(),256,[0,256]);


# hist=plt.hist(I.ravel(),256,[0,256]);

# w,h=I.shape
# s=w*h

# temp[0] y
# temp[1] x
# print(hist)
temp = 0
a = None
b = None
s = np.sum(hist[0])
for i in range(len(hist[0])):
    temp += hist[0][i] / s
    if temp > 0.05 and a is None:
        a = hist[1][i]
    elif temp > 0.95:
        b = hist[1][i]
        break
# print("a    ", a)
# print("b    ", b)

#crayfish
# a=100
# b=175

#office
# a=150
# b=200

#map
# a=160
# b=210

#train
# a=80
# b=230

#branches
# a=140
# b=220

#terrain
# a=160
# b=220

J = (I-a) * 255.0 / (b-a)
J[J < 0] = 0
J[J > 255] = 255
J = J.astype(np.uint8)

axes[0,1].imshow(J, 'gray', vmin=0, vmax=255)
axes[0,1].axis('off')

axes[1,1].hist(J.ravel(),256,[0,256]);

K = cv2.equalizeHist(I)

axes[0,2].imshow(K, 'gray', vmin=0, vmax=255)
axes[0,2].axis('off')

axes[1,2].hist(K.ravel(),256,[0,256]);
plt.show()