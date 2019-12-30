import os
from PIL import Image
#set path location
mypath = "C:/MyProject/"
for root,dirs,files in os.walk(mypath,topdown=False):
    for name in files:
        print(os.path.join(root, name))
        if os.path.splitext(os.path.join(root, name))[1].lower() == ".jpeg":
            if os.path.isfile(os.path.splitext(os.path.join(root, name))[0] + ".tiff"):
                print("A tiff file already exists for %s" % name)
            #if a tiff is *NOT* present,create one from the jpeg    
            else:
                outfile = os.path.splitext(os.path.join(root, name))[0] + ".tiff"
                try:
                    im = Image.open(os.path.join(root, name))
                    print ("Generating tiff for %s" % name)
                    im.thumbnail(im.size)
                    im.save(outfile, "TIFF", quality=100)
                except Exception as e:
                    print (e)
                    
                    
                                        
                    
                    