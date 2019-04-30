import urllib.request
import tarfile
import time
import os

import numpy as np
import cv2 as cv

from typing import List
from PIL import Image

from src.util import progress
from src.util import get_links_from_file

SHRINK = 'shrink'
CENTERING = 'centering'

class LocalReader():
    def __init__(self, height=256, width=256,
                 shaping=SHRINK, path='data/images/original')-> None:
        super().__init__()
        self.size = (width, height)
        self.shaping = shaping
        self.path = path

    def read(self, start=0, end=3)->np.ndarray:
        current=0
        for tar in os.listdir(self.path):
            if current == end:    
                break
            else:
                current += 1
                yield self.get_images(tar)
                
    def get_images(self, tar_name:str)->List:
        images = []
        with tarfile.open(self.path + "/" + tar_name, "r:gz") as tar:
            tar_size = len(tar.getmembers())
            item = 0
            for member in tar.getmembers():
                file = tar.extractfile(member)
                images.append(self.procces_image(file))
                progress(item, tar_size-1, status="{0}    ".format(tar_name))
                item += 1
            print()
        return np.array(images)
    
    def procces_image(self, file)->np.ndarray:
        image = np.asarray(bytearray(file.read()), dtype="uint8")
        image = cv.imdecode(image, cv.IMREAD_COLOR)
        if self.shaping == SHRINK:
            return cv.resize(image, self.size)
        elif self.shaping == CENTERING:
            #TODO: Finish Centring
            width = int(image.shape[0]/2) - int(self.size[0]/2)
            height = int(image.shape[1]/2) - int(self.size[1]/2)
            image = image[width:width+self.size[0], height:height+self.size[1]] 
            print(image.shape)
            return image