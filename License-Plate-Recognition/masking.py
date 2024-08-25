import numpy as np
import cv2

def mask(image: np.ndarray, points1: np.ndarray, cover: np.ndarray) -> np.ndarray:
    I1 = image.copy()
    I2 = cover.copy()
    points2 = np.array([(0,0),(I2.shape[1],0),(I2.shape[1],I2.shape[0]),(0,I2.shape[0])]).astype(np.float32)
    H = cv2.getPerspectiveTransform(points2, points1)
    output_size = (I1.shape[1], I1.shape[0])
    J = cv2.warpPerspective(I2, H, output_size)
    # for i in range(4):
    #     cv2.circle(I1, (int(points1[i, 0]), int(points1[i, 1])), 3, [0, 0, 255], 2)
    #     cv2.circle(I2, (int(points2[i, 0]), int(points2[i, 1])), 3, [0, 0, 255], 2)
    # cv2.imshow('I', I1)
    # cv2.waitKey(0)
    cv2.imshow('JI2', I2)
    cv2.waitKey(0)
    cv2.imshow('J', J)
    cv2.waitKey(0)
    # print(len(I1[0][0]))
    # print(len(I1[1]))
    # print(len(I1))
    counter=0
    for i in range(len(I1)):
        for j in range(len(I1[0])):
            # print(j)
            # print("l")
            # counter = counter+1
            # print(counter)
            black = [0,0,0]
            # print(J[i,j])
            # print(".........")
            if not(J[i,j] == black).all():
                # print('HI')
                I1[i,j]=J[i,j]


    cv2.imshow('s', I1)
    cv2.waitKey(0)
    return I1

# points = []
# im = cv2.imread('1.jpg')
# cv = cv2.imread('kntu.jpg')
# n,m,r=im.shape
# with open("1.txt", "r") as f2:
#     for line in f2:
#         point = line.split()
#         points.append(float(point[1])*m)
#         points.append(float(point[2])*n)
#         points.append(float(point[3])*m)
#         points.append(float(point[4])*n)
#         points.append(float(point[5])*m)
#         points.append(float(point[6])*n)
#         points.append(float(point[7])*m)
#         points.append(float(point[8])*n)
#
#
# pts = np.array([(points[0],points[1]),(points[2],points[3]),(points[4],points[5]),(points[6],points[7])]).astype(np.float32)
#
# mask(im,pts,cv)
