import Image
import os

dir = 'query_images/'
outfile = open("query_images.txt","w")

files = os.listdir(dir)
for file in files :
    im = Image.open(dir + file)
    name = file[0:10] + '.jpg'
    im.save(dir + name)
    outfile.write(dir + name + "\n")
