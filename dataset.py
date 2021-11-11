#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 10:15:51 2021

@author: roji
"""


from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np



class PersonAnimeDataset(Dataset):
    def __init__(self, root_anime, root_person, transform = None):
        
        self.root_anime = root_anime
        self.root_person = root_person
        self.transform = transform
        
        self.anime_images = os.listdir(root_anime)
        self.person_images = os.listdir(root_person)
        self.length_dataset = max(len(self.anime_images) , len(self.person_images))
        self.anime_len = len(self.anime_images)
        self.person_len = len(self.person_images)
        
    def __len__(self):
        return self.length_dataset
    
    def __getitem__(self,index):
        anime_img = self.anime_images[index % self.anime_len]
        person_img = self.person_images[index % self.person_len]
        
        anime_path = os.path.join(self.root_anime,anime_img)
        person_path = os.path.join(self.root_person,person_img)
        
        anime_img = np.array(Image.open(anime_path).convert("RGB"))
        person_img = np.array(Image.open(person_path).convert("RGB"))
        
        if self.transform :
            augmentations = self.transform(image = anime_img , image0 = person_img)
            anime_img = augmentations["image"]
            person_img = augmentations["image0"]
            
        return anime_img, person_img