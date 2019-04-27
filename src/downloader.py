import urllib.request
import logging
import tarfile
import time
import datetime
import sys
import os

import numpy as np

from requests_html import HTMLSession
from io import BytesIO
from typing import List
from PIL import Image

from src.util import progress
from src.util import get_link

class Downloader():
    def __init__(self, start:int=0, end:int=1)->None:
        self.start = start
        self.end = end
        self.iteration = 0
        self.links = []
        self.challenger = None
        self.path = None
        self.size = 0
        logging.basicConfig(filename='data/Downloader.log', level=logging.INFO)

    def download(self, path:str="data/AVA.txt")->None:
        self.path = path
        self.set_log_stamp()
        for links in self.get_links():
            self.create_dir()
            images = []
            download_size = len(links)
            self.size += download_size
            current_download = 0
            logging.info("START --- Iteration: {0}, Challenge: {1}, Tottal size:{2}"
                         .format(self.iteration, self.challenger, self.size))
            for link in links:
                progress(current_download, download_size-1, 
                         status="Downloading Iteration: {0}, Image id: {1}".format(self.iteration, link["id"]))
                images.append({
                    "id": link["id"],
                    "img":self.get_image(link["link"])
                })
                current_download += 1
            self.compress_images(images)
            logging.info("END --- Iteration: {0}, Challenge: {1}, Tottal size:{2}"
                        .format(self.iteration, self.challenger, self.size))
            self.iteration += 1
        

    def set_log_stamp(self)->None:
        logging.info("-------------------------------------------------------------------------------------------")
        logging.info("Run: {0}".format(datetime.datetime.fromtimestamp(time.time()).strftime('%d/%m/%Y %H:%M:%S')))

    def compress_images(self, img_map)->None:
        tar_name = "data/images/{0}/{1}.tar.gz".format(self.challenger, "original")
        current_image = 0
        image_list_size = len(img_map)
        if os.path.exists(tar_name):
            os.remove(tar_name)
        with tarfile.open(tar_name, "w:gz") as tar:
            for obj in img_map:
                progress(current_image, image_list_size-1, 
                         status="Process Iteration: {0}, Image id: {1}        ".format(self.iteration, obj["id"]))
                tarinfo = tarfile.TarInfo(name="{0}.jpg".format(obj["id"]))
                tarinfo.size = len(obj["img"])
                tar.addfile(tarinfo=tarinfo, fileobj=BytesIO(obj["img"]))
                current_image+=1
            
    def create_dir(self)->None:
        directory = "data/images/{0}".format(self.challenger)
        if not os.path.exists(directory):
            os.makedirs(directory)
        
    def get_links(self)->None:
        temp_challenger = None
        temp_links = []
        with open(self.path, "r") as file:
            for line in file:
                values = line.split(" ")
                img_id = values[1]
                challenger = values[14].rstrip()
                
                if temp_challenger == None:
                    temp_challenger = challenger
                
                if self.iteration >= self.start and self.iteration < self.end:
                    if temp_challenger == challenger:
                        temp_links.append({
                            "id": img_id,
                            "link":get_link(img_id, challenger)
                        })
                    elif temp_challenger != challenger:
                        self.links = temp_links
                        self.challenger = temp_challenger
                        temp_links = [{
                            "id": img_id,
                            "link":get_link(img_id, challenger)
                        }]
                        temp_challenger = challenger
                        yield self.links
                elif self.iteration >= self.end:
                    break
                else:
                    if temp_challenger != challenger:
                        temp_challenger = challenger
                        self.iteration += 1

    def get_image(self, link:str)->bytes:
        with urllib.request.urlopen(link) as response:
            return response.read()