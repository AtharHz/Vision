import cv2
import numpy as np
I = cv2.imread('damavand.jpg')
J = cv2.imread('eram.jpg')
for i in range(101):
    K = cv2.addWeighted(I, 1-(i/100), J, i/100, 0)
    cv2.imshow("Result", K)
    cv2.waitKey(90)
    # print(i)
cv2.destroyAllWindows()