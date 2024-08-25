import numpy as np
import cv2

# I = cv2.imread('branches2.jpg').astype(np.float32) / 255
I = cv2.imread('isfahan.jpg').astype(np.float32) / 255

noise_sigma = 0.04  # initial standard deviation of noise

m = 1  # initial filter size,

gm = 3  # gaussian filter size

size = 9  # bilateral filter size
sigmaColor = 0.3
sigmaSpace = 75

# with m = 1 the input image will not change
filter = 'b'  # box filter

while True:

    # add noise to image
    N = np.random.rand(*I.shape) * noise_sigma
    J = (I + N).astype(np.float32)

    if filter == 'b':
        # filter with a box filter
        K = cv2.blur(J, (m, m))
        # J = cv2.boxFilter(J,-1,(m,m))
        # pass
    elif filter == 'g':
        # filter with a Gaussian filter
        # Fg = cv2.getGaussianKernel(gm, sigma=-1)
        # Fg = Fg.dot(Fg.T)
        # J = cv2.filter2D(I, -1, Fg)
        K = cv2.GaussianBlur(J, (gm, gm), 0)
        # pass
    elif filter == 'l':
        # filter with a bilateral filter
        K = cv2.bilateralFilter(J, size, sigmaColor, sigmaSpace)
        # pass

    # filtered image
    cv2.imshow('img', K)
    key = cv2.waitKey(30) & 0xFF

    if key == ord('b'):
        filter = 'b'  # box filter
        print('Box filter')

    elif key == ord('g'):
        filter = 'g'  # filter with a Gaussian filter
        print('Gaussian filter')

    elif key == ord('l'):
        filter = 'l'  # filter with a bilateral filter
        print('Bilateral filter')

    elif key == ord('+'):
        # increase m
        m = m + 2
        print('m=', m)

    elif key == ord('-'):
        # decrease m
        if m >= 3:
            m = m - 2
        print('m=', m)
    elif key == ord('u'):
        # increase noise
        noise_sigma += 0.01
        print("noise_sigma= ", noise_sigma)
        # pass
    elif key == ord('d'):
        # decrease noise
        noise_sigma -= 0.01
        if noise_sigma < 0:
            noise_sigma = 0
        print("noise_sigma= ", noise_sigma)
        # pass
    elif key == ord('p'):
        # increase gm
        gm += 2
        print("gm= ", gm)
        # pass
    elif key == ord('n'):
        # decrease gm
        gm -= 2
        if gm < 1:
            gm=1
        print("gm= ", gm)
        # pass
    elif key == ord('>'):
        # increase size
        size += 2
        print("size= ", size)
        # pass
    elif key == ord('<'):
        # decrease size
        size -= 2
        if size < 1:
            size = 1
        print("size= ", size)
        # pass
    elif key == ord('q'):
        break  # quit

cv2.destroyAllWindows()