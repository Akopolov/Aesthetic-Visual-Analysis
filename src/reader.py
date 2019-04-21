import urllib.request

import numpy as np

from src.util import progress

from requests_html import HTMLSession
from typing import List
from PIL import Image

class Reader(HTMLSession):
    def __init__(self) -> None:
        super().__init__()
        self.base_url = "http://www.dpchallenge.com/image.php?IMAGE_ID="
        self.img_link_map = {}

    def read(self, txt_path:str=None, img_id_list:List[str]=None, img_id:str=None)-> np.ndarray:
        if txt_path != None:
            return self.read_from_txt(txt_path=txt_path)
        
        elif img_id_list != None:
            return self.read_from_list(img_id_list=img_id_list)
        
        elif img_id != None:
            return self.read_from_image_id(img_id=img_id)
    
    def read_from_txt(self, txt_path:str)-> np.ndarray:
#         progress(i+1, total, status='Downloading')
        pass
    
    def read_from_list(self, img_id_list:List[str])-> np.ndarray:
        total = len(img_id_list)
        for i in range(total):
            progress(i+1, total, status='Downloading images')
            pass
    
    def read_from_image_id(self, img_id:str)-> np.ndarray:
        return np.array([self.get_image(img_id=img_id)])
        
    def get_image(self, img_id:str)-> np.ndarray:
        img_link = None
        
        if img_id not in self.img_link_map:
            img_link = self.parse_html(img_id)
        else:
            img_link = self.img_link_map[img_id]

        with urllib.request.urlopen(img_link) as response:
            image = Image.open(response)
            return np.asarray(image)
    
    def parse_html(self, img_id:str)-> str:
        url = self.base_url + img_id
        response = self.get(url)
        img_container = response.html.find('#img_container', first=True)
        img_tag = img_container.find('img')[1]
        link = 'https:' + img_tag.attrs['src']
        
        if img_id not in self.img_link_map:
            self.img_link_map[img_id] = link
        
        return link