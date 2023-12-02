import torch as th
from torch.nn.functional import interpolate
import torch.nn as nn
from .io import *
import torchvision.transforms as T
import numpy as np




def gauss1d(sigma):
    r = int(np.ceil(sigma * 3))
    k = 1 + 2*r
    x = np.arange(k, dtype=np.float32) - r
    g = np.exp(-x**2 / (2*sigma**2))
    return (g / g.sum()).astype(np.float32)

def torch_gauss2d(img, sigma):
    img = img.permute(2,0,1)
    im_size = (img.shape[1], img.shape[2])
    nc = img.shape[0]
    img = img.unsqueeze(0)
    g = gauss1d(sigma)
    g = th.tensor(g, device=img.device).unsqueeze(0)
    g = th.stack([g]*nc, 0)
    img = img.transpose(2,3)
    img = img.reshape((1, 3, -1))
    c = th.nn.functional.conv1d(img, g, stride=1, groups=nc, padding='same')
    c = c.reshape(1, 3, im_size[1], im_size[0])
    c = c.transpose(2,3)
    c = c.reshape((1, 3, -1))
    c = th.nn.functional.conv1d(c, g, stride=1, groups=nc, padding='same')
    c = c.reshape(1, 3, im_size[0], im_size[1]).squeeze()
    c = c.permute(1,2,0)
    return c

def pyramid_loss(targ_img, curr_img, td_level: int = 8, sigma : float = 10.0):
    loss = 0.
    im_shape = targ_img.shape
    itd = 0
    count = 1
    totol_wight = 0.0
    weight = 1.0
    # print(targ_img.shape)
    targ_img = torch_gauss2d(targ_img, 16)
    curr_img = torch_gauss2d(curr_img, 16)

    loss += (targ_img - curr_img).abs().mean()

    targ_img = torch_gauss2d(targ_img, 32)
    curr_img = torch_gauss2d(curr_img, 32)

    loss += (targ_img - curr_img).abs().mean() * 2**2


    targ_img = torch_gauss2d(targ_img, 64)
    curr_img = torch_gauss2d(curr_img, 64)

    loss += (targ_img - curr_img).abs().mean() * 2**3

    targ_img = torch_gauss2d(targ_img, 128)
    curr_img = torch_gauss2d(curr_img, 128)

    loss += (targ_img - curr_img).abs().mean() * 2**4

    loss /= (1+2**2+2**3+2**4)



    # while True:
    #     if int(im_shape[1] / (itd+1)) < 10 or int(im_shape[0] / (itd+1)) < 10:
    #         break
    #     targ_pyd = T.Resize(size=(int(im_shape[1] / (itd+1)), int(im_shape[0] / (itd+1))))(targ_img.unsqueeze(0).permute(0,3,2,1))
    #     curr_pyd = T.Resize(size=(int(im_shape[1] / (itd+1)), int(im_shape[0] / (itd+1))))(curr_img.unsqueeze(0).permute(0,3,2,1))
    #     loss += (targ_pyd - curr_pyd).abs().mean() * (2**count)
    #     itd += td_level
    #     totol_wight +=  (2**count)
    #     count += 1
    # loss /= totol_wight
    return loss
