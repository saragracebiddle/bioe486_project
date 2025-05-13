import pandas as pd
import torch
import os
import numpy as np
from PIL import Image

def generate_target_fixed(image_id, annotation_file):
    # Read annotations
    data = pd.read_csv(annotation_file, sep=' ', header=None)
    data.columns = ['refno', 'tissue', 'class', 'severity', 'x', 'y', 'r']
    data['severity'] = data['severity'].fillna('N')

    # Clean refno
    data['refno_clean'] = data['refno'].str.extract(r'([a-zA-Z]*\d+)', expand=False)
    data['image_id'] = data['refno_clean'].str.extract(r'(\d+)', expand=False).astype(int) - 1

    # Filter by image_id
    rows = data.loc[data['image_id'] == image_id]

    if rows.empty:
        print(f"[WARNING] No annotation found for image ID {image_id}")
        target = {
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.zeros((0,), dtype=torch.int64),
            "image_id": torch.tensor([image_id])
        }
        return None, target

    imgPath = os.path.join('mias_data', rows['refno'].iloc[0] + '.pgm')
    img = Image.open(imgPath)
    w, h = img.size

    boxes = []
    labels = []

    for i in range(len(rows)):
        X = rows.iloc[i]['x']
        Y = rows.iloc[i]['y']
        R = rows.iloc[i]['r']

        if isinstance(X, str) and X == '*NOTE':
            X = w // 2
            Y = h // 2
            R = w // 4
        else:
            try:
                X = float(X)
                Y = float(Y)
                R = float(R)
            except:
                print(f"[WARNING] Invalid X, Y, R in row {i}, setting dummy box")
                X = w // 2
                Y = h // 2
                R = w // 4

        centerX = X / w
        centerY = Y / h
        startX = centerX - (R / w)
        startY = centerY - (R / h)
        endX = centerX + (R / w)
        endY = centerY + (R / h)

        if any(np.isnan([startX, startY, endX, endY])):
            print(f"[ERROR] NaN found in box calculation for image {image_id}. Skipping.")
            continue

        boxes.append([startX, startY, endX, endY])

        severity = rows.iloc[i]['severity']
        if severity == 'B':
            labels.append(1)
        elif severity == 'M':
            labels.append(2)
        else:
            labels.append(0)

    target = {
        "boxes": torch.as_tensor(boxes, dtype=torch.float32),
        "labels": torch.as_tensor(labels, dtype=torch.int64),
        "image_id": torch.tensor([image_id])
    }

    return img, target

# Direct test block
if __name__ == "__main__":
    img, target = generate_target_fixed(3, 'mias_info/labels.txt')
    print(f"Boxes: {target['boxes']}")
    print(f"Labels: {target['labels']}")
