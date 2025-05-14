from torch.utils.data import Dataset
from plotting import generate_target
import os
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from plotting import generate_box, generate_label

def get_label(rows, r):
    label = rows['severity'].iloc[r]
    if label == "B":
        return 1
    elif label == "M":
        return 2
    else:
        return 0

def get_target(image_name, rows, index):
        
    imgPath = os.path.join('mias_data', rows['refno'].iloc[0] + '.pgm')
    with open(imgPath, 'rb') as pgmf:
        im = plt.imread(pgmf)

    img = np.array(im, dtype = np.float32)
    img /= img.max()     

    # Bounding boxes for objects
    # In coco format, bbox = [xmin, ymin, width, height]
    # In pytorch, the input should be [xmin, ymin, xmax, ymax]
    boxes = []
    labels = []
    areas = []
    for r in range(len(rows['refno'])):
        (xmin, ymin, xmax,  ymax) = generate_box(rows, r, img)

        boxes.append([xmin, ymin, xmax, ymax])
        label = get_label(rows, r)
        labels.append(label)
        area = (ymax - ymin) * (xmax - xmin)
        areas.append(area)
        
    
    boxes = torch.as_tensor(boxes, dtype=torch.float32)
    
    # Labels (In my case, I only one class: target class or background)
    labels = torch.as_tensor(labels, dtype=torch.int64)
    # Tensorise img_id
    index = torch.tensor([index])
    areas = torch.as_tensor(areas, dtype = torch.float32)
    # Annotation is in dictionary format
    target = {}
    target["boxes"] = boxes
    target["labels"] = labels
    target["image_id"] = index
    target['area'] = areas

    if np.isnan((target['boxes']).numpy()).any() or target['boxes'].shape == torch.Size([0]):
        target['boxes'] = torch.zeros((0,4),dtype=torch.float32)

    return img, target


# dataset class
class customDataset(Dataset):
    # initialize the constructor
    def __init__(self, imgs = None, transforms=None):
        self.transforms = transforms
        self.imgs = imgs

    def __getitem__(self, index):

        refno = self.imgs[index]

        name, type = refno.split('.')

        file = 'mias_info/labels.txt'
        with open(file) as f:
            data = pd.read_csv(f,  sep = ' ', header = None)
            data.columns = ['refno', 'tissue', 'class', 'severity','x','y','r']
            data['severity'] = data['severity'].fillna('N')
            data['image_id'] = data.refno.str.split('b', expand = True)[1].astype(int) -1

            rows = data.loc[data['refno'] == name]
        
        image, target = get_target(refno, rows, index)
    
        image = np.stack([np.array(image)] * 3, axis=-1)
        h, w = image.shape[:2]

        boxes = target["boxes"]
        
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.numpy()

        if boxes.shape[0] > 0:
            boxes[:, [0, 2]] *= w
            boxes[:, [1, 3]] *= h
        boxes = boxes.tolist()
        labels = target["labels"].tolist()
    
        if self.transforms is not None:
            transformed = self.transforms(
                image=image,
                bboxes=boxes,
                labels=labels
            )
            image = transformed["image"]
            boxes = torch.tensor(transformed["bboxes"], dtype=torch.float32)
            labels = torch.tensor(transformed["labels"], dtype=torch.int64)
            
        target["boxes"] = boxes
        if np.isnan((target['boxes']).numpy()).any() or target['boxes'].shape == torch.Size([0]):
            target['boxes'] = torch.zeros((0,4),dtype=torch.float32)
        target["labels"] = labels
        #image = image.astype('float32') / image.max()

        return image, target

    def __len__(self):
        # return the size of the dataset
        return len(self.imgs)	