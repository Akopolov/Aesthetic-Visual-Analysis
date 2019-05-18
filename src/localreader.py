import urllib.request
import tarfile
import time
import os

import numpy as np
import cv2 as cv

from typing import List, Dict, Tuple
from PIL import Image
from io import BytesIO

from src.util import progress
from src.util import get_links_from_file

SHRINK = 'shrink'
CENTERING = 'centering'

class LocalReader():
    def __init__(self, height=256, width=256, validation_size=0.1 ,shaping=SHRINK, 
                 img_path='data/images/original', ava_path='data/AVA.txt')-> None:
        super().__init__()
        self.validation_size = validation_size
        self.size = (width, height)
        self.shaping = shaping
        self.img_path = img_path
        self.ava_path = ava_path
        self.img_save_path = None

    def train(self, start=0, end=3)->Tuple:
        for iteration, tar in enumerate(os.listdir(self.img_path)):
            if iteration == end:    
                break
            elif iteration < start:
                iteration += 1
                continue
            else:
                iteration += 1
                yield self.get_images(tar_name=tar, isValidation=False)
                
    def validate(self, start=0, end=3)->Tuple:
        for iteration, tar in enumerate(os.listdir(self.img_path)):
            if iteration == end:    
                break
            elif iteration < start:
                iteration += 1
                continue
            else:
                iteration += 1
                yield self.get_images(tar_name=tar, isValidation=True)
                
    def preprocess(self, start=0, end=3, img_save_path='data/images/')->None:
        self.img_save_path = img_save_path + self.shaping
        
        if not os.path.exists(self.img_save_path):
            os.makedirs(self.img_save_path)
            
        for iteration, tar in enumerate(os.listdir(self.img_path)):
            temp_validation_size = self.validation_size
            self.validation_size = 0.0
            if iteration == end:    
                break
            elif iteration < start:
                iteration += 1
                continue
            else:
                iteration += 1
                print("Iteration: {0}, {1}".format(iteration, tar))
                self.save_images(tar_name=tar)
        self.validation_size = temp_validation_size
        
    def save_images(self, tar_name)->None:
        name_img_map = {}
        old_tar_path = self.img_path + "/" + tar_name
        new_tar_path = self.img_save_path + "/" + tar_name
        
        if os.path.exists(new_tar_path):
            os.remove(new_tar_path)
            
        with tarfile.open(old_tar_path, "r:gz") as tar:
            for member in tar.getmembers():
                file = tar.extractfile(member)
                img = np.asarray(bytearray(file.read()), dtype="uint8")
                name_img_map[member.name] = self.procces_image(img)

        with tarfile.open(new_tar_path, "w:gz") as tar:
            for key in name_img_map.keys():
                img = BytesIO()
                Image.fromarray(name_img_map[key]).save(img, format='JPEG')
                tarinfo = tarfile.TarInfo(name="{0}".format(key))
                tarinfo.size = len(img.getvalue())
                tar.addfile(tarinfo=tarinfo, fileobj=BytesIO(img.getvalue()))
                
    def get_images(self, tar_name:str, isValidation:bool)->Tuple:
        images = []
        labels = []
        label_map = {}
        with tarfile.open(self.img_path + "/" + tar_name, "r:gz") as tar:
            label_map = self.get_scores(tar_name.split(".")[0])
            tar_size = len(tar.getmembers())

            start = 0
            end = 0
            
            if(isValidation):
                start = int(tar_size * (1 - self.validation_size))
                end = tar_size
            else:
                start = 0
                end = int(tar_size * (1 - self.validation_size))
                
            for count, member in enumerate(tar.getmembers()):
                if(count>= start and count<end):                    
                    file = tar.extractfile(member)
                    labels.append(label_map[member.name.split(".")[0]])
                    img = np.asarray(bytearray(file.read()), dtype="uint8")
                    images.append(self.procces_image(img))
                
        return (np.array(images), np.array(getMean(labels)))
    
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