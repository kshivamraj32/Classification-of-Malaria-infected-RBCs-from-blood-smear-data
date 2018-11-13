from collections import OrderedDict
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu, threshold_adaptive, rank
from skimage.morphology import watershed, disk
from scipy import ndimage
import SimpleITK as sitk
import numpy as np
import cv2
from skimage.segmentation import clear_border
from skimage.color import label2rgb
from radiomics import featureextractor
import six
from scipy.spatial import distance
import math
import matplotlib.pyplot as plt  # Plotting package
import copy
import csv
import os
from pathlib import Path
import re

def edge_enhance(img):
    #generating the kernels
    kernel = np.array([[-1,-1,-1,-1,-1],
                               [-1,2,2,2,-1],
                               [-1,2,8,2,-1],
                               [-2,2,2,2,-1],
                               [-1,-1,-1,-1,-1]])/8.0

    #process and output the image
    output = cv2.filter2D(img, -1, kernel)
    return output

def doImgProc(path, spath, extractor):
    print("Starting Image Processing")
    csvfile = spath+"/featuredataset.csv"

    ORGIMG = cv2.imread(path)
    ORGIMG2 = ORGIMG.copy()
    ORGIMG3 = ORGIMG.copy()
    #cv2.imshow("Input Original Image", ORGIMG)
    shifted = cv2.pyrMeanShiftFiltering(ORGIMG, 21, 51)
    #cv2.imshow("Mean Shift Filtered Image", shifted)
    sharped = edge_enhance(shifted)
    #cv2.imshow("Edge Sharped Image", sharped)
    gray = cv2.cvtColor(sharped, cv2.COLOR_BGR2GRAY)
    th, im_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    #cv2.imshow("Thresholded Binary Image", im_thresh)
    im_bordercleared = clear_border(im_thresh)
    #cv2.imshow("Thresholded Binary Image", im_bordercleared)
    im_floodfill = im_bordercleared.copy()
    h, w = im_bordercleared.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255);
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    fill_image = im_bordercleared | im_floodfill_inv
    #cv2.imshow("FloodFilled Binary Image", fill_image)
    kernel = np.ones((5, 5), dtype=np.uint8)
    binary_cleaned = cv2.morphologyEx(fill_image, cv2.MORPH_OPEN, kernel)
    binary_mask = binary_cleaned
    binary_mask = cv2.bitwise_not(binary_mask)
    #cv2.imshow('binary mask',binary_mask)
    #cv2.waitKey(0)
    #cv2.imshow("Cleaned Binary Image", binary_cleaned)
    #cv2.waitKey(0)


    binary_dt = ndimage.distance_transform_edt(binary_cleaned)
    binary_dt = ndimage.filters.maximum_filter(binary_dt, size=2)
    #plt.imshow(binary_dt, interpolation='none', cmap='gray')
    #plt.show()

    localMax = peak_local_max(binary_dt, indices=False, min_distance=10, labels=binary_cleaned)
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-binary_dt, markers, mask=binary_cleaned)
    #print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))
    image_label_overlay = label2rgb(labels, image=binary_cleaned)
    #cv2.imshow("Labelled Image", image_label_overlay)
    stat = True

    for label in np.unique(labels):
        if label == 0:
            continue
        mask = np.zeros(gray.shape, dtype="uint8")
        mask[labels == label] = 255
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
        c = max(cnts, key=cv2.contourArea)
        ((x, y), r) = cv2.minEnclosingCircle(c)
        #print('centre : %f,%f'%(x,y))
        img_shape=(np.array(ORGIMG3)).shape
        rows=img_shape[0]
        col=img_shape[1]
        if r < 25:
            continue
        if int(y-r)<0 or int(y+r)>rows or int(x-r)<0 or int(x+r)>col:
            continue
        col_image_patch = ORGIMG3[(int)(y-r):(int)(y+r), (int)(x-r):(int)(x+r)]
        col_image_patch2 = col_image_patch.copy()
        bin_image_patch = binary_mask[(int)(y - r):(int)(y + r), (int)(x - r):(int)(x + r)]
        bin_image_patch = cv2.bitwise_not(bin_image_patch)
        #cv2.imshow('bin_image_patch',bin_image_patch)
        #cv2.waitKey(0)
        bin_image_patch2 = bin_image_patch.copy()
        cip=(np.array(col_image_patch)).shape
        #print('shape of color image patch: %d,%d'%(cip[0],cip[1]))
        #print('LABEL: %f ; CENTER: %f , %f ; RADIUS: %f' % (label, r, r, r))
        height = cip[0]
        width = cip[1]
        #dd = min(width, height)
        #radius = dd/2.0
        for i in range(0, height):
           for j in range(0, width):
                dst = math.sqrt(math.pow((r - j), 2) + math.pow((r - i), 2))
                if dst > r:
                    col_image_patch2[i, j] = (0, 0, 0)
                    bin_image_patch2[i, j] = 0

        cv2.circle(ORGIMG, (int(x), int(y)), int(r), (0, 255, 0), 2)
        cv2.rectangle(ORGIMG2, ((int)(x - r), (int)(y - r)), ((int)(x + r), (int)(y + r)), (0, 255, 0), 3)
        cv2.putText(ORGIMG, "#{}".format(label), (int(x) - 10, int(y)),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imwrite(spath+"/col%d_1.png" % (label), col_image_patch)
        cv2.imwrite(spath+"/col%d_2.png" % (label), col_image_patch2)
        cv2.imwrite(spath+"/bin%d_1.png" % (label), bin_image_patch)
        cv2.imwrite(spath+"/bin%d_2.png" % (label), bin_image_patch2)
        cv2.imwrite(spath + "/labelled.png", ORGIMG)
        norm_col_image = doCellSizeNormalize(spath+"/col%d_1.png" % (label),spath+"/col%d_2.png" % (label))
        norm_bin_image = doCellSizeNormalize(spath+"/bin%d_1.png" % (label),spath+"/bin%d_2.png" % (label))
        cv2.imwrite(spath + "/norm_col%d.png" % (label), norm_col_image)
        cv2.imwrite(spath + "/norm_bin%d.png" % (label), norm_bin_image)
        gray_norm_bin = cv2.cvtColor(norm_bin_image, cv2.COLOR_BGR2GRAY)
        ret, norm_bin = cv2.threshold(gray_norm_bin, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        FEATUREDATAVECTOR,FEATURENAMES = doCellFeatExtract(norm_col_image, norm_bin, extractor,stat)
        #FEATUREDATAVECTOR, FEATURENAMES = doCellFeatExtract(col_image_patch2, bin_image_patch2, extractor, stat)
        FEATUREDATAVECTOR=np.append([label],FEATUREDATAVECTOR,axis=0)
        #print(FEATUREDATAVECTOR.shape)
        if (stat):
            FEATURENAMES = np.append(['Feature Name'], FEATURENAMES, axis=0)
            a=np.array([FEATURENAMES])
            a = np.append(a,[FEATUREDATAVECTOR],axis=0)
            stat = False
        else:
            a = np.append(a, [FEATUREDATAVECTOR], axis=0)
        #print(FEATUREDATAVECTOR)
        #with open(csvfile, "w") as output:
         #   writer = csv.writer(output, lineterminator='\n')
          #  for val in FEATUREDATAVECTOR:
           #     writer.writerow([val])

    with open(csvfile, "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        writer.writerows(a)

        # show the output image
    #cv2.imshow("Output", ORGIMG)
    #cv2.imshow("Output2", ORGIMG2)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

def doCellFeatExtract(cell_image, mask_image, extractor,stat):
    gray_image_patch = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)
    im_arr = np.array(gray_image_patch);
    im_arr = np.expand_dims(im_arr, axis=0)
    image3d = sitk.GetImageFromArray(im_arr)
    im_arr = np.array(mask_image);
    im_arr = np.expand_dims(im_arr, axis=0)
    mask3d = sitk.GetImageFromArray(im_arr)
    usresult = extractor.execute(image3d, mask3d)

    #Sort dictionary according to keys
    result = OrderedDict(sorted(usresult.items()))
    cellFeat = []
    featName=[]
    count = 0

    for key, val in six.iteritems(result):
        #print("\t%s: %s" % (key, val))
        if count > 11:
            cellFeat.append(val)
            if (stat):
                featName.append(key)

        count = count+1

    #print(featName)
    return (cellFeat,featName)

def doCellSizeNormalize(path1, path2):
    img = np.array(cv2.imread(path1))
    img2 = np.array(cv2.imread(path2))
    [height, width, channel] = img.shape
    RED = img[:, :, 2]
    GREEN = img[:, :, 1]
    BLUE = img[:, :, 0]
    # cv2.imshow('BLUE',BLUE)
    # cv2.waitKey(0)
    CHA = [BLUE, GREEN, RED]
    MASKPIX = []
    MASK_W = 224
    MASK_H = 224
    for i in range(0, 3):
        boundary_pixels = []
        ch = CHA[i]
        frs = ch[0, 🙂
        lrs = ch[height - 1, 🙂
        fcs = ch[:, 0]
        lcs = ch[:, width - 1]
        r = np.append(frs, lrs, axis=0)
        c = np.append(fcs, lcs, axis=0)
        boundary_pixels = np.append(r, c, axis=0)
        #print(len(boundary_pixels))
        boundary_pixels = np.sort(boundary_pixels)
        med = np.median(boundary_pixels, axis=0)
        MASKPIX = np.append(MASKPIX, [(int)(med)], axis=0)
        #print(MASKPIX)

    MASKIMG = np.zeros((MASK_H, MASK_W, 3), dtype=np.uint8)
    MASKIMG[np.where((MASKIMG == [0, 0, 0]).all(axis=2))] = MASKPIX
    img[np.where((img2 == [0, 0, 0]).all(axis=2))] = MASKPIX

    # cv2.imshow('final',MASKIMG)
    # cv2.waitKey(0)
    cm = [(int)(MASK_H / 2), (int)(MASK_W / 2)]
    cp = [(int)(height / 2), (int)(width / 2)]
    top_left = [cm[0] - cp[0], cm[0] - cp[1]]
    #print(top_left)
    start_row = top_left[0]
    start_col = top_left[1]
    MASKIMG[start_row:start_row + height, start_col:start_col + width] = img
    # MASKIMG[np.where((MASKIMG==[0,0,0]).all(axis=2))] = MASKPIX
    #cv2.imshow('superimposed1', MASKIMG)
    #cv2.waitKey(0)
    return MASKIMG

def main():
    path = '/home/spmlgpuone/SPMLMEMBERS/maitreya/data/malimages'
    params = '/home/spmlgpuone/SPMLMEMBERS/maitreya/data/features/pyradiomics/examples/exampleSettings/exampleCT.yaml'
    rx = re.compile(r'\.(jpg)')
    for path, dnames, fnames in os.walk(path):
        for x in fnames:
            if rx.search(x):
                imgpath = os.path.join(path, x)
                imgname = os.path.splitext(os.path.basename(imgpath))[0]
                imgparent = Path(imgpath).parent
                dirpath = "%s/%s" % (imgparent, imgname)
                print("Processing...%s"%imgpath)
                Path(dirpath).mkdir(parents=True, exist_ok=True)
                extractor = featureextractor.RadiomicsFeaturesExtractor(params)
                doImgProc(imgpath, dirpath, extractor)


if _name_ == '_main_':
    main()
