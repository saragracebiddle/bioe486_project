# import the necessary packages
from torch.utils.data import Dataset
from plotting import generate_target
import os
import config
import numpy as np
import torch

def list_files_walk(start_path='.'):
    out = []

    for root, dirs, files in os.walk(start_path):
        for file in files:
            out.append(file)


    return out

class CustomTensorDataset(Dataset):
    # initialize the constructor
    def __init__(self, transforms=None):
        #self.tensors = tensors
        self.transforms = transforms
        self.imgs = list_files_walk('mias_data/')

    def __getitem__(self, index):
        # grab the image, label, and its bounding box coordinates
        #image = self.tensors[0][index]
        #label = self.tensors[1][index]
        #bbox = self.tensors[2][index]

        # transpose the image such that its channel dimension becomes
        # the leading one
        #image = image.permute(2, 0, 1)
        # check to see if we have any image transformations to apply
        # and if so, apply them
        
        image, target = generate_target(index, 'mias_info/labels.txt')
    
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
        target["labels"] = labels
    
        return image, target

    def __len__(self):
        # return the size of the dataset
        return len(self.imgs)	