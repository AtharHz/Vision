import numpy as np
import cv2

def extract(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    m = 450
    n = 100
    output_size = (m,n)
    points2 = np.array([(0, 0), (m, 0), (m, n), (0, n)]).astype(np.float32)
    H = cv2.getPerspectiveTransform(points, points2)
    J = cv2.warpPerspective(image, H, output_size)
    # for i in range(4):
    #     cv2.circle(image, (int(points[i, 0]), int(points[i, 1])), 3, [0, 0, 255], 2)
    #     cv2.circle(image, (int(points2[i, 0]), int(points2[i, 1])), 7, [0, 0, 255], 2)
    # cv2.imshow('a', image)
    # cv2.waitKey(0)
    cv2.imshow('J', J)
    cv2.waitKey(0)
    return J










# points = []
# im = cv2.imread('4.jpg')
# n,m,r=im.shape
# with open("4.txt", "r") as f2:
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
# extract(im,pts)