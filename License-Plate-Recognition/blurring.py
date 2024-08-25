import numpy as np
import cv2

from extract import extract
from masking import mask

def blur(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    plate = extract(image, points)
    # plate = cv2.boxFilter(plate, -1, (17, 17))
    plate = cv2.GaussianBlur(plate, (35, 35), 0)
    output = mask(image, points, plate)
    return output

points = []
im = cv2.imread('4.jpg')
n, m, r = im.shape
with open("4.txt", "r") as f2:
    for line in f2:
        point = line.split()
        points.append(float(point[1]) * m)
        points.append(float(point[2]) * n)
        points.append(float(point[3]) * m)
        points.append(float(point[4]) * n)
        points.append(float(point[5]) * m)
        points.append(float(point[6]) * n)
        points.append(float(point[7]) * m)
        points.append(float(point[8]) * n)

pts = np.array([(points[0], points[1]), (points[2], points[3]), (points[4], points[5]), (points[6], points[7])]).astype(
    np.float32)

# print(im.shape[1])
# print(im.shape[0])
blur(im, pts)

