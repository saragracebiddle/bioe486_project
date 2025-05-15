import cv2 as cv
import numpy as np
from skimage.morphology import erosion, dilation

def denoise(img):
    # first threshold the image
    ret, threshold = cv.threshold(img, 20, 255, cv.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8) 
    eroded = cv.erode(threshold, kernel, iterations = 1)
    dilated = cv.dilate(eroded, kernel, iterations = 1)
    # connected component analysis to find the largest component
    analysis = cv.connectedComponentsWithStats(dilated, 4, cv.CV_32S)
    (totalLabels, label_ids, values, centroid) = analysis
    # 
    output = np.zeros(threshold.shape, dtype=np.uint8)

    areas = values[:, cv.CC_STAT_AREA]
    # the first connected componenet with index 0 is the background
    # which we will ignore
    largest = np.argmax(areas[1:])
    # add one to largest since the first index was skipped
    componentMask = (label_ids == (largest+1)).astype(np.uint8) *255
    # create mask
    mask = cv.bitwise_or(output, componentMask)

    tfmask = np.equal(mask, 255)
    np.copyto(output, img, where = tfmask)

    return output


def denoise_all(stack):
    output = stack.copy()
    for i, img in enumerate(stack):
        denoised = denoise(img)
        output[i] = denoised
    
    return output