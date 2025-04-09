import os, sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from segmentation_mask_overlay import overlay_masks
import matplotlib.colors as mcolors

def createOverlay3(imagePath, maskPath, predPath, resultsPath):
    # image = Image.open(imagePath).convert("L")
    # image = np.array(image)

    image = plt.imread(imagePath)
    mask = plt.imread(maskPath)
    pred = plt.imread(predPath)
    image = (image / np.max(image) * 255).astype('uint8')
    mask = (mask / np.max(mask) * 255).astype('uint8') > 127
    pred = (pred / np.max(pred) * 255).astype('uint8') > 127

    masks = [mask, pred]
    # [Optional] prepare labels
    mask_labels = ['GT', 'Pred']

    # [Optional] prepare colors
    cmap = plt.cm.tab20(np.arange(len(mask_labels)))
    colors = [mcolors.CSS4_COLORS['lawngreen'], mcolors.CSS4_COLORS['red']]
    #colors = [mcolors.CSS4_COLORS['lawngreen']]
    # Laminate your image!
    fig = overlay_masks(image, masks, labels=mask_labels, colors=colors, mask_alpha=0.5)

    # Do with that image whatever you want to do.
    fig.savefig(resultsPath, bbox_inches="tight", dpi=300)



patient = '8ac4645595'
sliceList = [38, 76, 119, 208]
for sliceNumber in sliceList:
    sliceNumber = str.zfill(str(sliceNumber), 3)
    imagePath = f'/home/ricardo/Desktop/results/AUTOMI_PTV_tot_DiceLoss/overlays_{patient}/{patient}_{sliceNumber}.png'
    predPath = f'/home/ricardo/Desktop/results/AUTOMI_PTV_tot_DiceLoss/overlays_{patient}/{patient}_{sliceNumber}_mask.png'
    maskPath = f'/home/ricardo/Desktop/results/AUTOMI_PTV_tot_DiceLoss/overlays_{patient}/{patient}_{sliceNumber}_target.png'
    resultsPath = f'/home/ricardo/Desktop/results/AUTOMI_PTV_tot_DiceLoss/overlays_{patient}/{patient}_{sliceNumber}_superposition_gt.png'
    createOverlay3(imagePath, maskPath, predPath, resultsPath)