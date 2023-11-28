import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

import torch
from ivt.transform import *
import numpy as np
import igl
import cv2

def linear_to_srgb(l):
    if l <= 0.00313066844250063:
        return l * 12.92
    else:
        return 1.055*(l**(1.0/2.4))-0.055

def srgb_to_linear(s):
    if s <= 0.0404482362771082:
        return s / 12.92
    else:
        return ((s+0.055)/1.055) ** 2.4

def to_srgb(image):
    return np.clip(np.vectorize(linear_to_srgb)(to_numpy(image)), 0, 1)

def to_linear(image):
    return np.vectorize(srgb_to_linear)(to_numpy(image))

def to_numpy(data):
    if torch.is_tensor(data):
        return data.detach().cpu().numpy()
    else:
        return data

def range_loss(V, weight, left, right):
    lossV = torch.pow(torch.where(V < left, left - V, torch.where(V > right, V - right, torch.zeros_like(V))), 2)
    loss = (lossV.mean()) * weight
    return loss

def load_mask(mask_path, erode_val = 0):
    mat_mask = cv2.imread(str(mask_path))
    mat_mask = cv2.erode(mat_mask, np.ones((erode_val, erode_val), np.uint8))
    # mat_mask = cv2.resize(mat_mask, (int(res[0]),int(res[1])), interpolation = cv2.INTER_NEAREST)
    mat_mask[mat_mask[:,:,0]>0.001] = 1.0
    mat_mask[mat_mask[:,:,0]<0.001] = 0.0
    mat_mask = mat_mask.astype(np.float32)
    return mat_mask


def read_obj(obj_path):
    obj_path = str(obj_path)
    
    v, tc, n, f, ftc, fn = igl.read_obj(obj_path)
    
    return v, tc, n, f, ftc, fn

    



from torch.nn.functional import interpolate
import torch as th
import torch.nn as nn


def find_contours(img):
    kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
    morphed  = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    contours = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours[-2]

def threshold(img, thresh=128, maxval=255, type=cv2.THRESH_BINARY):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    threshed = cv2.threshold(img, thresh, maxval, type)[1]
    return threshed

def find_contours(img):
    kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
    morphed  = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    contours = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours[-2]

def max_contour(contours):
    return sorted(contours, key=cv2.contourArea)[-1]

def mask_from_contours(ref_img, contours):
    mask = np.zeros(ref_img.shape, np.uint8)
    mask = cv2.drawContours(mask, contours, -1, (255,255,255), -1)
    return cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

def dilate_mask(mask, kernel_size=11):
    kernel  = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=1)
    return dilated

def smooth_mask(mask, kernel_size=11):
    blurred  = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)
    threshed = threshold(blurred)
    return threshed


def write_contours_exr(data):
    # print("mask sum:", data.sum())
    if len(data.shape) == 4:
        data = data[0]
    if torch.is_tensor(data):
        data = data.cpu().numpy()
    img = ((cv2.cvtColor(data, 1, cv2.COLOR_RGB2BGR).astype(np.uint8))*255 * -1 + 255).astype(np.uint8)
    threshed = threshold(img, type=cv2.THRESH_BINARY_INV)
    contours = find_contours(threshed)
    mask        = mask_from_contours(img, contours)
    contours = find_contours(mask)
    hull = []
    for i in range(len(contours)):
        hull.append(cv2.convexHull(contours[i], False))
    # create a mask from hull points
    # print(hull)
    hull = mask_from_contours(img, hull).astype(np.float32) / 255

    return hull

    print(hull)
    exit()
    xmin = 0
    ymin = 0

    x_sum = np.sum(hull, axis=0)
    y_sum = np.sum(hull, axis=1)

    xmax = len(x_sum)-1
    ymax = len(y_sum)-1

    for xx in range(len(x_sum)):
        if x_sum[xx] > 0:
            xmin = xx
            break
    for yy in range(len(y_sum)):
        if y_sum[yy] > 0:
            ymin = yy
            break

    for xx in range(len(x_sum)-1, -1, -1):
        if x_sum[xx] > 0:
            xmax = xx
            break

    for yy in range(len(y_sum)-1, -1, -1):
        if y_sum[yy] > 0:
            ymax = yy
            break

    # print(xmin, ymin, xmax, ymax)
    # print(hull.shape)
    # exit()
    # cv2.imwrite(fname, hull.astype(np.float32))
    if not crop:
        if toFile:
            cv2.imwrite(fname, hull.astype(np.float32))

        return hull.astype(np.float32)
    cropped_hull = hull[ymin:ymax, xmin:xmax]
    # print(cropped_hull)
    if len(cropped_hull) == 0:
        return 99999999
    # cv2.imwrite(fname, cropped_hull.astype(np.float32))

    old_size = (cropped_hull.shape[:2]) # old_size is in (height, width) format
    # print(old_size)
    if max(old_size) == 0:
        return 99999999
    desired_size = data.shape[0]

    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    # new_size should be in (width, height) format
    # print(len(cropped_hull))
    # print("here", cropped_hull)
    try:
        im = cv2.resize(cropped_hull, (new_size[1], new_size[0]))
    except:
        return 99999999
    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=color).astype(np.float32)
    if toFile:
        cv2.imwrite(fname, new_im)
    return new_im
    # return hull.sum()



def write_obj(obj_path, v, f, tc=None, ftc=None):
    if len(f.shape) == 1:
        if not torch.is_tensor(f):
            f = torch.tensor(f)
        f = f.unsqueeze(0)
    obj_file = open(obj_path, 'w')

    def f2s(f):
        return [str(e) for e in f]

    v = to_numpy(v).astype(float)
    f = to_numpy(f).astype(int) + 1

    if tc is None and ftc is None:
        for v_ in v:
            obj_file.write(f"v {' '.join(f2s(v_))}\n")
        for f_ in f:
            obj_file.write(f"f {' '.join(f2s(f_))}\n")
    else:
        tc = to_numpy(tc).astype(float)
        ftc = to_numpy(ftc).astype(int)

        if tc.size > 0 and ftc.size == f.size:
            ftc += 1
            for v_ in v:
                obj_file.write(f"v {' '.join(f2s(v_))}\n")
            for tc_ in tc:
                obj_file.write(f"vt {' '.join(f2s(tc_))}\n")
            for f_, ftc_ in zip(f, ftc):
                obj_file.write(f"f {f_[0]}/{ftc_[0]} {f_[1]}/{ftc_[1]} {f_[2]}/{ftc_[2]}\n")
        else:
            for v_ in v:
                obj_file.write(f"v {' '.join(f2s(v_))}\n")
            for f_ in f:
                obj_file.write(f"f {' '.join(f2s(f_))}\n")

    obj_file.close()
