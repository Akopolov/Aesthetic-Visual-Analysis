import urllib.request
import tarfile
import time
import os

import numpy as np
import cv2 as cv

from typing import List, Dict, Tuple
from PIL import Image

from src.util import progress
from src.util import get_links_from_file

SHRINK = 'shrink'
CENTERING = 'centering'

class LocalReader():
    def __init__(self, height=256, width=256, shaping=SHRINK, 
                 img_path='data/images/original', ava_path='data/AVA.txt')-> None:
        super().__init__()
        self.size = (width, height)
        self.shaping = shaping
        self.img_path = img_path
        self.ava_path = ava_path

    def read(self, start=0, end=3)->Tuple:
        current=0
        for tar in os.listdir(self.img_path):
            if current == end:    
                break
            elif current < start:
                current += 1
                continue
            else:
                current += 1
                yield self.get_images(tar)
                
    def get_images(self, tar_name:str)->Tuple:
        images = []
        labels = []
        label_map = {}
        with tarfile.open(self.img_path + "/" + tar_name, "r:gz") as tar:
            label_map = self.get_scores(tar_name.split(".")[0])
            tar_size = len(tar.getmembers())
            item = 0
            for member in tar.getmembers():
                file = tar.extractfile(member)
                labels.append(label_map[member.name.split(".")[0]])
                img = np.asarray(bytearray(file.read()), dtype="uint8")
                images.append(self.procces_image(img))
                progress(item, tar_size-1, status="{0}    ".format(tar_name))
                item += 1
            print()
        return (np.array(images), np.array(labels))
    
    def get_scores(self, challenge:str)->Dict:
        can_break = False
        score_map = {}
        with open(self.ava_path, "r") as file:
            for line in file:
                values = line.split(" ")
                if challenge == values[14].rstrip():
                    score_map[values[1]] = list(map(int, values[2:12]))
                    can_break = True
                elif can_break:
                    return score_map
            return score_map
    
    def procces_image(self, img)->np.ndarray:
        img = cv.imdecode(img, cv.IMREAD_COLOR)
        if self.shaping == SHRINK:
            return cv.resize(img, self.size)
        elif self.shaping == CENTERING:
            #TODO: Finish Centring
            print(img.shape)
            width = int(img.shape[0]/2) - int(self.size[0]/2)
            height = int(img.shape[1]/2) - int(self.size[1]/2)
            img = img[width:width+self.size[0], height:height+self.size[1]] 
            print(img.shape)
            return img

def mean_function(labels):
    result = 0
    for i in range(len(labels)):
        result += (i + 1) * labels[i]
    result /= np.sum(labels)
    return result
        
def getMean(scores):
    return np.array(list(map(mean_function, scores)), dtype=np.float64)