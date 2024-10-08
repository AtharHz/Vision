import cv2
import numpy as np

NO_CORNERS = 78

def first_correct_winsize(I):
    "find the smallest win_size for which all corners are detected"
    # write your code here
    I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    I = np.float32(I)
    for i in reversed(range(0,8)):
        print("i", i)
        win_size = 2 ** i
        sobel_kernel_size = 3  # kernel size for gradients
        alpha = 0.04
        H = cv2.cornerHarris(I, win_size, sobel_kernel_size, alpha)
        H = H / H.max()
        C = np.uint8(H > 0.01) * 255
        nc, CC = cv2.connectedComponents(C)
        # print("ws",win_size)
        # print("nc",nc)
        if (nc-1) == NO_CORNERS:
            return win_size


I1 = cv2.imread('kntu1.jpg')
I2 = cv2.imread('kntu4.jpg')

s1 = first_correct_winsize(I1)
s2 = first_correct_winsize(I2)
    
J = np.concatenate((I1, I2), 1)

if s1 < s2:
    txt = 'Logo 1 is %d times smaller than logo 2'%(s2/s1)
elif s1 > s2:
    txt = 'Logo 1 is %d times larger than logo 2'%(s1/s2)
else:
    txt = 'Logo 1 is about the same size as logo 2'
    
cv2.putText(J,txt,(20,40), \
                cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2)

cv2.imshow('scale',J)
cv2.waitKey(0)
    
    
