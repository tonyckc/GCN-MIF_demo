# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 10:44:28 2020

@author: ckc
"""
import numpy as np
import os
import random
from PIL import Image
LDCT_path = "/hdd/chenkecheng/CT/LDCT_six_number/"
NDCT_path = "/hdd/chenkecheng/CT/NDCT_six_number/"
SAVE_path  ="/hdd/chenkecheng/CT/LDCT_patch_train/"
SAVE_path2  ="/hdd/chenkecheng/CT/NDCT_patch_train/"
j= 0
'''
for i in os.listdir(NDCT_path):
    im = Image.open(os.path.join(NDCT_path,i))
    im.load()
    im = np.asarray(im,dtype="float32")
    im = Image.fromarray(np.asarray(im,dtype='uint8'))
    im.save("F:/dataset-CT/L67_image/NDCT/NDCT_six_number/{}.tif".format(j))
    j+=1
    print(j)
'''
image_id = random.sample(range(0,3411),3411)
patch_image_id = random.sample(range(0,218304),218304)

j= 0
number=1
for i  in image_id:
    im_LDCT = '{}.tif'.format(i)
    im_NDCT ='{}.tif'.format(i)
    # load image
    im_L = Image.open(os.path.join(LDCT_path,im_LDCT))
    im_L.load()
    im_L1 = np.asarray(im_L,dtype="float32")
    im_N = Image.open(os.path.join(NDCT_path,im_NDCT))
    im_N.load()
    im_N1 = np.asarray(im_N,dtype="float32")
    
    
    # get patch
    x_point = random.sample(range(0,512-64),64)
    y_point = random.sample(range(0,512-64),64)
    for p in range(64):
        patch_L = im_L1[x_point[p]:(x_point[p]+64),y_point[p]:(y_point[p]+64)]
        patch_N = im_N1[x_point[p]:(x_point[p]+64),y_point[p]:(y_point[p]+64)]
        im_L2 = Image.fromarray(np.asarray(patch_L,dtype="uint8"))
        im_N2 = Image.fromarray(np.asarray(patch_N,dtype="uint8"))
        im_L2.save(os.path.join(SAVE_path,"{}.tif".format(patch_image_id[j])))
        im_N2.save(os.path.join(SAVE_path2,"{}.tif".format(patch_image_id[j])))
        print(number)
        number+=1
        j+=1
        