import tarfile
import time
import os

import numpy as np

from typing import List, Dict, Tuple
from PIL import Image
from io import BytesIO

SHRINK = 'shrink'
CENTERING = 'centering'

class ProcessedReader():
    def __init__(self, validation_size=0.1 ,shaping=SHRINK, 
                 img_path='data/images/', ava_path='data/AVA.txt')-> None:
        super().__init__()
        self.validation_size = validation_size
        self.shaping = shaping
        self.img_path = img_path + shaping
        self.ava_path = ava_path

    def train(self, start=0, end=3)->Tuple:
        current=0
        for tar in os.listdir(self.img_path):
            if current == end:    
                break
            elif current < start:
                current += 1
                continue
            else:
                current += 1
                yield self.get_images(tar_name=tar, isValidation=False)
                
    def validate(self, start=0, end=3)->Tuple:
        current=0
        for tar in os.listdir(self.img_path):
            if current == end:    
                break
            elif current < start:
                current += 1
                continue
            else:
                current += 1
                yield self.get_images(tar_name=tar, isValidation=True)
                
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
                    img = Image.open(BytesIO(np.asarray(bytearray(file.read()), dtype="uint8")))
                    images.append(np.array(img))
                
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

def mean_function(labels):
    result = 0
    for i in range(len(labels)):
        result += (i + 1) * labels[i]
    result /= np.sum(labels)
    return result
        
def getMean(scores):
    return np.array(list(map(mean_function, scores)), dtype=np.float64)