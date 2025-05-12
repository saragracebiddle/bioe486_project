import numpy as np
import config
import os
import cv2

def preprocessing(info):
    labels = []
    data = []
    bboxes = []
    imgPaths = []

    for i in range(len(info['refno'])):
        filename = info['refno'][i] + '.png'
        label = info['severity'][i]
        # derive the path to the input image, load the image (in
		# OpenCV format), and grab its dimensions
        imagePath = os.path.sep.join([config.IMAGES_PATH, label,
			filename])
        image = cv2.imread(imagePath)
        (h, w) = image.shape[:2]

        X = info['x'][i]
        Y = info['y'][i]
        R = info['r'][i]
        if X == '*NOTE':
            X = w //2
            Y = h //2
            R = w // 4
        else:
            X = float(X)
            Y = float(Y)
            R = float(R)
        
        centerX = float(X) / w
        centerY = float(Y) / h
        startX = centerX - (float(R)/w)
        startY = centerY - (float(R/h))
        endX = centerX + (float(R)/w)
        endY = centerY + (float(R)/h)

        data.append(image)
        labels.append(label)
        bboxes.append((startX, startY, endX, endY))
        imgPaths.append(imagePath)

    data = np.array(data, dtype="float32")
    labels = np.array(labels)
    bboxes = np.array(bboxes, dtype="float32")
    imgPaths = np.array(imgPaths)
    

    return data, labels, bboxes, imgPaths

