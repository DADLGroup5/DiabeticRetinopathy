import numpy as np
import os
import imageio    # writing images 
from PIL import Image

name_dictionary = {}  #for grade of eye
f=open("Location of data ")

contents=f.read()
contents=contents.split('\n')     #splitting the name 

for i in range(len(contents)-1):
  contents[i]=contents[i].split(',')     #split by the comma (,)
  name_dictionary[contents[i][0]]=contents[i][1]        


#counting dict for stage

stage=name_dictionary.values()
stage_set=set(stage)   #converting iteratable in distinct sequence using set function
counting_dictionary= {}    #empty dictionary 

for i in stage_set:
    counting_dictionary[i]=0

#now mapping images with stage number

for i in os.listdir('F:/data'):
    imageName=i.split('.')[0]   #splitting on basis of .
    label=name_dictionary[str(imageName)]
    counting_dictionary[label]+= 1
    path=os.path.join('F:/data', i)
    saveName='./labeled_train/' + label + '-' + str(counting_dictionary[label])+ '.tif'
    
    image_data=np.array(Image.open(path))
    imageio.imwrite(saveName,image_data)
    
    
  
    
    
counting_dictionary