import numpy as np
import cv2
import glob
from PIL import Image



for name in glob.glob('../datasets/test/val/gt/0/*.png'):
        im = Image.open(name)
        pixels = im.load() # create the pixel map
        
        for i in range(im.size[0]): # for every pixel:
            for j in range(im.size[1]):
              if pixels[i,j] == (1):
                      pixels[i,j] = (255)
        im.save(name, 'PNG')

