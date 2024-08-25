import numpy as np
import cv2
from matplotlib import pyplot as plt

def std_filter(I, ksize):
    F = np.ones((ksize,ksize), dtype=np.float) / (ksize*ksize);
    MI = cv2.filter2D(I,-1,F) # apply mean filter on I
    I2 = I * I; # I squared
    MI2 = cv2.filter2D(I2,-1,F) # apply mean filter on I2
    return np.sqrt(abs(MI2 - MI * MI))
def zero_crossing(I):
    """Finds locations at which zero-crossing occurs, used for
    Laplacian edge detector"""
    Ishrx = I.copy();
    Ishrx[:,1:] = Ishrx[:,:-1]
    Ishdy = I.copy();
    Ishdy[1:,:] = Ishdy[:-1,:]
    ZC = (I==0) | (I * Ishrx < 0) | (I * Ishdy < 0); # zero crossing locations
    SI = std_filter(I, 3) / I.max()
    Mask = ZC & (SI > .1)
    E = Mask.astype(np.uint8) * 255 # the edges
    return E

cam_id = 0  # camera id

lable_text='original'

# for default webcam, cam_id is usually 0
# try out other numbers (1,2,..) if this does not work

cap = cv2.VideoCapture(cam_id)

mode = 'o' # show the original image at the beginning

sigma = 5

while True:

    ret, I = cap.read();
    # I = cv2.imread("agha-bozorg.jpg") # can use this for testing
    I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY) # convert to grayscale
    Ib = cv2.GaussianBlur(I, (sigma,sigma), 0); # blur the image
    
    if mode == 'o':
        # J = the original image
        J = I
        lable_text = 'original'
    elif mode == 'x':
        # J = Sobel gradient in x direction
        J = np.abs(cv2.Sobel(Ib,cv2.CV_64F,1,0));
        lable_text = 'Sobel X'
    elif mode == 'y':
        # J = Sobel gradient in y direction
        J = np.abs(cv2.Sobel(Ib, cv2.CV_64F, 0, 1));
        lable_text = 'Sobel Y'
        # pass

    
    elif mode == 'm':
        # J = magnitude of Sobel gradient
        Ix = cv2.Sobel(Ib, cv2.CV_64F, 1, 0)
        Iy = cv2.Sobel(Ib, cv2.CV_64F, 0, 1)
        J = np.sqrt(Ix * Ix + Iy * Iy)
        lable_text = 'Sobel gradient'
        # pass
    
    elif mode == 's':
        # J = Sobel + thresholding edge detection
        Ix = cv2.Sobel(Ib, cv2.CV_64F, 1, 0)
        Iy = cv2.Sobel(Ib, cv2.CV_64F, 0, 1)
        Es = np.sqrt(Ix * Ix + Iy * Iy)
        J = np.uint8(Es > 90) * 255
        lable_text = 'Sobel thresholding'
        # pass
    
    elif mode == 'l':
        # J = Laplacian edges
        Ib = cv2.GaussianBlur(I, (sigma, sigma), 0);
        El = cv2.Laplacian(Ib, cv2.CV_64F, ksize=5)
        J = zero_crossing(El);
        lable_text = 'Laplacian edges'
        # pass

    
    elif mode == 'c':
        # J = Canny edges
        Ib = cv2.GaussianBlur(I, (sigma, sigma), 0);
        J = cv2.Canny(Ib, 50, 120)
        lable_text = 'Canny edges'
        # pass
    
    # we set the image type to float and the
    # maximum value to 1 (for a better illustration)
    # notice that imshow in opencv does not automatically
    # map the min and max values to black and white. 
    J = J.astype(np.float_) / J.max();
    J = cv2.putText(J, lable_text , (10, 40) , cv2.FONT_HERSHEY_SIMPLEX,1.0,(125, 246, 55))
    cv2.imshow("my stream", J);

    key = chr(cv2.waitKey(1) & 0xFF)

    if key in ['o', 'x', 'y', 'm', 's', 'c', 'l']:
        mode = key
    if key == '-' and sigma > 1:
        sigma -= 2
        print("sigma = %d"%sigma)
    if key in ['+','=']:
        sigma += 2    
        print("sigma = %d"%sigma)
    elif key == 'q':
        break

cap.release()
cv2.destroyAllWindows()
