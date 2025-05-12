from bs4 import BeautifulSoup
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import os
import cv2
from PIL import Image
import numpy as np


def generate_box(info, index, img):
    (h, w) = np.array(img).shape[:2]

    X = info['x'].iloc[index]
    Y = info['y'].iloc[index]
    R = info['r'].iloc[index]
    if X == '*NOTE':
        X = w //2
        Y = h //2
        R = w // 4
    else:
        X = float(X)
        Y = h - float(Y)
        R = float(R)
        
    centerX = float(X) / w
    centerY = float(Y) / h
    startX = centerX - (float(R)/w)
    startY = centerY - (float(R/h))
    endX = centerX + (float(R)/w)
    endY = centerY + (float(R)/h)
    
    return [startX, startY, endX, endY]

def generate_label(rows, r):
    label = rows['severity'].iloc[0]
    if label == "B":
        return 1
    elif label == "M":
        return 2
    return 0

def generate_target(image_id, file):

    with open(file) as f:
        data = pd.read_csv(f,  sep = ' ', header = None)
        data.columns = ['refno', 'tissue', 'class', 'severity','x','y','r']
        data['severity'] = data['severity'].fillna('N')
        data['image_id'] = data.refno.str.split('b', expand = True)[1].astype(int) -1

        rows = data.loc[data['image_id'] == image_id]

        imgPath = os.path.join('mias_data', rows['refno'].iloc[0] + '.pgm')
        with open(imgPath, 'rb') as pgmf:
            im = plt.imread(pgmf)

        img = np.array(im, dtype = np.uint8)
        img = Image.fromarray(img.astype('uint8'), 'L')


        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        boxes = []
        labels = []
        for r in range(len(rows['refno'])):
            boxes.append(generate_box(rows, r,img))
            labels.append(generate_label(rows, r))
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # Labels (In my case, I only one class: target class or background)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        # Tensorise img_id
        img_id = torch.tensor([image_id])
        # Annotation is in dictionary format
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = img_id

        if np.isnan((target['boxes']).numpy()).any() or target['boxes'].shape == torch.Size([0]):
            target['boxes'] = torch.zeros((0,4),dtype=torch.float32)
        
        return img, target
    
def plot_image(img_tensor, annotation):
    
    fig,ax = plt.subplots(1)
    img = img_tensor.cpu().data

    # Display the image
    ax.imshow(img.permute(1, 2, 0))

    c, h,w = img.shape
    
    for box in annotation["boxes"]:
        xmin, ymin, xmax, ymax = box
        xmin, ymin, xmax, ymax = int(xmin * w), int(ymin * h), int(xmax * w), int(ymax * h)

        # Create a Rectangle patch
        rect = patches.Rectangle((xmin,ymin),(xmax-xmin),(ymax-ymin),linewidth=1,edgecolor='r',facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)

    plt.show()
