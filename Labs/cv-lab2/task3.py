import cv2
import numpy as np
I = cv2.imread('damavand.jpg')
J=I.copy()
B = I[:,:,0]
G = I[:,:,1]
R = I[:,:,2]
# I[:,:,1]=I[:,:,0]
# I[:,:,2]=I[:,:,0]
# I[:,:,0]=I[:,:,1]
# I[:,:,2]=I[:,:,1]
# I[:,:,0]=I[:,:,2]
# I[:,:,1]=I[:,:,2]
# J[:,:,0] = J[:,:,1] = J[:,:,2] =B//3+G//3+R//3
# temp = B//3+G//3+R//3
# J = np.repeat(temp[:, :, np.newaxis], 3, axis=2)
J[:,:,0] = B//3+G//3+R//3
J[:,:,1] = B//3+G//3+R//3
J[:,:,2] = B//3+G//3+R//3
# gray_image = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
# J[:,:,0]=gray_image
# J[:,:,1]=gray_image
# J[:,:,2]=gray_image
# cv2.imshow("Result", J)
# cv2.waitKey(10000)
for i in range(101):
    K = cv2.addWeighted(J, 1-(i/100), I, i/100, 0)
    cv2.imshow("Result", K)
    cv2.waitKey(100)
    # print(i)
cv2.destroyAllWindows()