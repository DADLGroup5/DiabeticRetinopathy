# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 20:49:57 2019

@author: Karthikeyan S
"""
from statistics import median
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import skimage.segmentation as seg
from PIL import Image

def circle_points(resolution, center, radius):
    """
    Generate points which define a circle on an image.Centre refers to the centre of the circle
    """   
    radians = np.linspace(0, 2*np.pi, resolution)
    c = center[1] + radius*np.cos(radians)#polar co-ordinates
    r = center[0] + radius*np.sin(radians)
    
    return np.array([c, r]).T
# Exclude last point because a closed path should not have duplicate points

x_centre, y_centre = 0,0
x_list = [0]
y_list = [0]
pix_list = [0]
#Read and display image in grayscale
image =  Image.open('diabetic_eye.tif').convert('L')
fig, ax= plt.subplots(nrows=1,ncols=1,figsize=(20,20))
plt.imshow(image)
plt.colorbar()

for x , y in zip(range(image.size[0]),range(image.size[1])):
    pix = image.getpixel((x,y))
    pix_list.append(pix)

        
print(max(pix_list))
max_pix = max(pix_list)
for x , y in zip(range(image.size[0]),range(image.size[1])):
    if (max_pix == image.getpixel((x,y))):
        x_centre = x
        y_centre = y
    
radius = 100
points = circle_points(200, [x_centre, y_centre], radius) 
ax.plot(points[:, 0], points[:, 1], '--r', lw=3) 

plt.savefig('output.png')


