import numpy as np
import config
import os
import cv2

def preprocessing(info):
    labels = []
    data = []
    bboxes = []
    imgPaths = []

    ls = os.listdir(config.IMAGES_PATH)
    for label in ls:
        files = os.listdir(os.path.sep.join([config.IMAGES_PATH, label]))
        for f in files:
            imgPath = os.path.sep.join([config.IMAGES_PATH, label, f])
            image = cv2.imread(imgPath)
            (h, w) = image.shape[:2]
            
            name, type = f.split('.')
            rows = info.loc[info['refno'] == name]

            bbox = []
            for r in range(len(rows['refno'])):
                X = info['x'].iloc[r]
                Y = info['y'].iloc[r]
                R = info['r'].iloc[r]
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

                bbox.append((startX, startY, endX, endY))

            data.append(image)
            labels.append(label)
            bboxes.append(bbox)
            imgPaths.append(imgPath)



    data = np.array(data, dtype="float32")
    #labels = np.array(labels)
    #bboxes = np.array(bboxes, dtype="float32")
    imgPaths = np.array(imgPaths)
    

    return data, labels, bboxes, imgPaths

