from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset

import os
import numpy as np
import cv2
import pandas as pd
from PIL import Image, ImageDraw
import modules.utils as utils

    
class Dataset_tu(BaseDataset):
    CLASSES = ['hair']
    
    def __init__(
            self, 
            mode = 'train',
            classes=None, 
            augmentation=None, 
            preprocessing=None,
    ):
        self.data_dir = '데이터 ROOT PATH'

        if mode == 'train':
            path = os.path.join(self.data_dir, '훈련 데이터 PATH')
        elif mode =='val':
            path = os.path.join(self.data_dir, 'VALIDATION 데이터 PATH')
        else:
            path = os.path.join(self.data_dir, 'TEST 데이터 PATH')


        if not os.path.isdir(os.path.join(path, 'masks')):
            os.makedirs(path+'masks')
        self.labels_fps = os.path.join(path, 'labels')
        self.labels_fps = [os.path.join(path, i) for i in self.labels_fps]
        self.polygon_to_mask(path)

        self.images_fps = os.listdir(os.path.join(path, 'images'))
        self.images_fps = [os.path.join(path, 'images', i) for i in self.images_fps]
        self.file_name = os.listdir(os.path.join(path, 'masks'))
        self.masks_fps = [os.path.join(path, 'masks', i) for i in self.file_name]

        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)
        
        masks = [(mask == 255)]
        mask = np.stack(masks, axis=-1).astype('float')

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
        
    def __len__(self):
        return len(self.images_fps)

    def polygon_to_mask(self, path):
        # os.makedirs(os.path.join(path,'masks'))
        mask_dir = os.path.join(path,'masks')
        for file_path in os.listdir(os.path.join(path,'images')):
            label_file = open(os.path.join(path,'labels',file_path))
            polygon = []
            for line in label_file:
                x,y = line.split(' ')
                x,y = float(x), float(y)
                polygon.append((x,y))
            json_file = utils.load_json(os.path.join(path,'labels.json'))

            polygon_dict = json_file['polygon1']
            polygon = []
            
            for point in polygon_dict:
                polygon.append(tuple(point.values()))
            
            img = Image.new('L', (512, 512), 'black')
            ImageDraw.Draw(img).polygon(polygon, outline='white', fill='white')
            img.save(os.path.join(mask_dir,file_path.split('.')[0]+'.jpg'))


if __name__=='__main__':
    a = Dataset_tu(mode= 'train')