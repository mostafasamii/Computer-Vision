import numpy as np
import pickle
import cv2
import os
from matplotlib import pyplot as plt
import pandas as pd
import argparse
import time
import cv2

def Read_Images():
    All_images = []
    ImagesPath = 'Images/'
    for img in os.listdir(ImagesPath):
        path = os.path.join(ImagesPath+img)
        All_images.append(cv2.imread(path))
    return All_images

Gunimages = Read_Images()
configPath = 'yolov3_testing.cfg'
LablesPath = 'yolov3.txt'

f=open(LablesPath, "r")
#Lables = f.read().strip().split('\n')
Lables = ["Gun"]

weightsPath = 'yolov3_.weights'

net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

np.random.seed(42)
#COLORS = np.random.randint(0, 255, size=(len(Lables), 3),dtype="uint8")
COLORS = np.random.uniform(0, 255, size=(len(Lables), 3))

boxes = []
output_confidences = []
classIDs = []

for img in Gunimages:
    start = time.time()
    height, width, channels = img.shape
    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(ln)
    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                # Object detected
                print(class_id)

                '''
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                '''
                box = detection[0:4] * np.array([width, height, width, height])
                (centerX, centerY, width, height) = box.astype("int")
                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                output_confidences.append(float(confidence))
                class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in range(len(boxes)):
        if i in indexes:
            #x, y, w, h = boxes[i]
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            label = str(Lables[class_ids[i]])
            color = COLORS[class_ids[i]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 3, color, 2)

    end = time.time()
    print("[INFO] YOLO took {:.6f} seconds".format(end - start))
    #cv2.imshow("Image", img)
    #key = cv2.waitKey(0)
    plt.figure(figsize=(20, 20))
    plt.grid()
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

print(output_confidences)
cv2.destroyAllWindows()
