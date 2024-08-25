import numpy as np
from ultralytics import YOLO
import cv2
import math
from sort import *

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("b.mp4")
cap.set(4, 720)


model = YOLO("../yolo-weights/yolov8n.pt")

ClassName = ['person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light',
             'fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow',
             'elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee',
             'skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard',
             'tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple',
             'sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch',
             'potted plant','bed','dining table','toilet','tv','laptop','mouse','remote','keyboard',
             'cell phone','microwave','oven','toaster','sink','refrigerator',
             'book','clock','vase','scissors','teddy bear','hair drier', 'toothbrush']

tracker = Sort(max_age=20)

limits = [100,550,700,500]
totalCount = []

while True:
    success, image = cap.read()
    image = cv2.resize(image, (1000, 700))
    results = model(image, stream=True)

    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:

            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)


            #Confidence
            conf = math.ceil((box.conf[0]*100))/100


            #Class Name
            cls =int(box.cls[0])
            currentClass = ClassName[cls]
            if currentClass == 'car' or currentClass == 'truck' or currentClass == 'bus' and conf > 0.8:
                cv2.putText(image, str(conf), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 3)
                cv2.putText(image, str(ClassName[cls]), (x1 + 41, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    resultTracker = tracker.update(detections)
    cv2.line(image,(limits[0],limits[1]),(limits[2],limits[3]),(0,0,255),5)

    for result in resultTracker:
        x1, y1, x2, y2, Id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2-x1, y2-y1
        cv2.putText(image, str(Id), (x1+72, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cx, cy = x1+w//2, y1+h//2
        cv2.circle(image,(cx,cy),5,(255, 0, 255), cv2.FILLED)
        if limits[0]< cx < limits[2] and limits[3]-20 < cy < limits[1]+20:
            print("id ", Id)
            if totalCount.count(Id) == 0:
                totalCount.append(Id)
        print(result)
        # print("count: ",len(totalCount))
    cv2.putText(image, "count: "+str(len(totalCount)), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)
    cv2.imshow("Image", image)
    cv2.waitKey(100)
