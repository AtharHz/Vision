import cv2
import numpy as np

I = cv2.imread('polygons.jpg')
G = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)

ret, T = cv2.threshold(G, 220, 255, cv2.THRESH_BINARY_INV)

nc1, CC1 = cv2.connectedComponents(T)

for k in range(1, nc1):

    Ck = np.zeros(T.shape, dtype=np.float32)
    Ck[CC1 == k] = 1
    Ck = cv2.GaussianBlur(Ck, (5, 5), 0)
    Ck = cv2.cvtColor(Ck, cv2.COLOR_GRAY2BGR)

    # Now, apply corner detection on Ck
    Ck = cv2.cvtColor(Ck, cv2.COLOR_BGR2GRAY)
    Ck = np.float32(Ck)
    window_size = 7
    sobel_kernel_size = 3  # kernel size for gradients
    alpha = 0.04
    H = cv2.cornerHarris(Ck, window_size, sobel_kernel_size, alpha)
    H = H / H.max()
    C = np.uint8(H > 0.01) * 255
    nC, CC, stats, centroids = cv2.connectedComponentsWithStats(C)
    J = I.copy()
    # print("It has %d corners")
    cv2.imshow('corners', Ck)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(J, 'There are %d vertices!' % (nC - 1), (20, 30), font, 1, (0, 0, 255), 1)
    for i in range(1, nC):
        cv2.circle(J, (int(centroids[i, 0]), int(centroids[i, 1])), 3, (0, 0, 255))
    cv2.imshow('corners', J)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

    # cv2.imshow('corners', Ck)
    # cv2.waitKey(0)  # press any key
