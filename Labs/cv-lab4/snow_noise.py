import numpy as np
import cv2

I = cv2.imread('isfahan.jpg', cv2.IMREAD_GRAYSCALE);
I = I.astype(np.float_) / 255

sigma = 0.04 # initial standard deviation of noise
# N = np.random.randn(*I.shape) * sigma

while True:

    N = np.random.randn(*I.shape) * sigma
    J = I + N # change this line so J is the noisy image
    
    cv2.imshow('snow noise',J)
    
    # press any key to exit
    key = cv2.waitKey(33)
    if key & 0xFF == ord('u'):  # if 'u' is pressed
        sigma += 0.01
        # notice maximum intensity is 1
        if sigma > 1:
            sigma = 1
        # N = np.random.randn(*I.shape) * sigma
        # pass # increase noise
    elif key & 0xFF == ord('d'):  # if 'd' is pressed
        sigma -= 0.01
        if sigma < 0:
            sigma = 0
        # N = np.random.randn(*I.shape) * sigma
        # pass # decrease noise
    elif key & 0xFF == ord('q'):  # if 'q' is pressed then 
        break# quit

    
cv2.destroyAllWindows()
