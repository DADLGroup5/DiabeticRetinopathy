import os
def main():
   i = 0
   #setting path location
   path="C:/MyProject/"
   for filename in os.listdir(path):
      my_dest ="moderate-" + str(i) + ".tiff"
      my_source =path + filename
      my_dest =path + my_dest
      #rename() function will rename all the files
      os.rename(my_source, my_dest)
      i += 1
if __name__ == '__main__':
   # Calling main() function
   main()
   
   
  
  
  