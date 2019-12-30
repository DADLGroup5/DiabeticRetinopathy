# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 21:38:50 2019

@author: Karthikeyan S
"""

from matplotlib import pyplot as plt 
from PIL import Image
import numpy as np
  
# reads an input image 
image =  Image.open('diabetic_eye.jpg').convert('L')
  
histogram, bin_edges = np.histogram(image, bins=10)
# find frequency of pixels in range 0-255 
plt.figure()
plt.title("Eye affected by diabetic retinopathy")
plt.xlabel("Grayscale Value")
plt.ylabel("Pixels") 
plt.plot(bin_edges[0:-1], histogram)  
plt.savefig('unhealthy.png')
plt.show()

# reads an input image 
image =  Image.open('healthy_eye.tif').convert('L')
  
histogram, bin_edges = np.histogram(image, bins=10)
# find frequency of pixels in range 0-255 
plt.figure()
plt.title("Healthy eye")
plt.xlabel("Grayscale Value")
plt.ylabel("Pixels") 
plt.plot(bin_edges[0:-1], histogram)  
plt.savefig('healthy.png')
plt.show()
