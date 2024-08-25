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
    window_size = 5
    sobel_kernel_size = 3  # kernel size for gradients
    alpha = 0.04
    H = cv2.cornerHarris(Ck, window_size, sobel_kernel_size, alpha)
    H = H / H.max()
    # C = np.uint8(H > 0.01) * 255
    # C[C < 0.01] = 0
    # print(len(C))
    # print(len(C[0]))
    J = I.copy()
    D = H.copy()
    cv2.imshow('corners', Ck)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break
    for i in range(len(H)):
        for j in range(len(H[0])):
            if H[i][j] > 0.01:
                adjacent = np.array([H[i-1, j-1], H[i-1, j], H[i-1, j+1],
                                    H[i, j-1], H[i, j], H[i, j+1],
                                    H[i+1, j-1], H[i+1, j], H[i+1, j+1]])
                if H[i][j] == np.max(adjacent):
                    D[i][j] = 1
                else:
                    D[i][j] = 0
            else:
                D[i][j] = 0
                    # print(adjacent.max())
                    # if C[i][j] != adjacent.max():
                    #     # print("whst")
                    #     C[i][j] = 0
                    # else:
                    #     # cv2.circle(J, (i, j), 3, (0, 0, 255))
                    #     C[i-1][j-1] = 0
                    #     C[i+1][j+1] = 0
                    #     C[i-1][j+1] = 0
                    #     C[i+1][j-1] = 0
                    #     C[i][j+1] = 0
                    #     C[i][j-1] = 0
                    #     C[i-1][j] = 0
                    #     C[i+1][j] = 0
    for i in range(len(D)):
        for j in range(len(D[0])):
            if D[i][j] != 0:
                cv2.circle(J, (j, i), 3, (0, 0, 255))
    print("Corners : ", np.count_nonzero(D))
    # J[C != 0] = (0, 0, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(J, 'There are %d vertices!' %np.count_nonzero(D), (20, 30), font, 1, (0, 0, 255), 1)
    cv2.imshow('corners', J)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break
    # cv2.imshow('corners', Ck)
    # cv2.waitKey(0)  # press any key
