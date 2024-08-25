import cv2
import numpy as np
I = cv2.imread('damavand.jpg')
J = cv2.imread('eram.jpg')
print(I.shape)
print(J.shape)
# K = I.copy()
# K[::2,::2,:] = J[::2,::2,:]
K = I//2+J//2
# K = np.clip((0.8*I + 0.2*J), 0, 255).astype(np.uint8)
# K = cv2.addWeighted(I,0.8,J,0.2, 0)
# K = cv2.addWeighted(I,0.1,J,0.9, 0)
# K = cv2.addWeighted(I,0.3,J,0.7, 0)
cv2.imshow("Image 1", I)
cv2.imshow("Image 2", J)
cv2.imshow("Blending", K)
cv2.waitKey(10000)
cv2.destroyAllWindows()