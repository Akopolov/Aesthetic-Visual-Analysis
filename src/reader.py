import urllib.request
import time

import numpy as np
import cv2 as cv

from requests_html import HTMLSession
from typing import List
from PIL import Image

from src.util import progress
from src.util import get_links_from_file

class Reader(HTMLSession):
    def __init__(self, img_width=256, img_height=256, batches=5, epochs=2)-> None:
        super().__init__()
        self.base_url = "http://www.dpchallenge.com/image.php?IMAGE_ID="
        self.img_width = img_width
        self.img_height = img_height
        self.batches = batches
        self.epochs = epochs

    def read(self, path:str=None, img_id:str=None)-> np.ndarray:
        if path != None:
            return self.read_from_txt(path=path)
        
        elif img_id != None:
            return self.read_from_image_id(img_id=img_id)
    
    def read_from_txt(self, path:str)-> np.ndarray:
        for epoch in range(self.epochs):
            images = []
            item = 0
            img_links = get_links_from_file(path=path, batches=self.batches, epoch=epoch)
            start_time = time.time()
            for link in img_links:
                image = self.get_image_from_link(link=link)
                images.append(image)
                item += 1
                progress(item, self.batches, status="Downloading Images")
            print("\nDownloading time: {0}s".format(round(time.time() - start_time, 2)))
            yield np.array(images)

    
    def read_from_image_id(self, img_id:str)-> np.ndarray:
        return np.array([self.get_image_by_id(img_id=img_id)])
    
    def get_image_by_id(self, img_id:str)-> np.ndarray:
        link = self.parse_html(img_id)
        return self.get_image_from_link(link=link)
    
    def get_image_from_link(self, link:str)-> np.ndarray:
        with urllib.request.urlopen(link) as response:
            img = np.asarray(bytearray(response.read()), dtype="uint8")
            img = cv.imdecode(img, cv.IMREAD_COLOR)
            img = cv.resize(img, (self.img_width, self.img_height))
            return np.asarray(img)
        
    def parse_html(self, img_id:str)-> str:
        url = self.base_url + img_id
        response = self.get(url)
        img_container = response.html.find('#img_container', first=True)
        img_tag = img_container.find('img')[1]
        link = 'https:' + img_tag.attrs['src']
        
        if img_id not in self.img_link_map:
            self.img_link_map[img_id] = link
        
        return link