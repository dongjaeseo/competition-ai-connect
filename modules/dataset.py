from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset

import os
import numpy as np
import cv2
import pandas as pd
from PIL import Image, ImageDraw
import modules.utils as utils
import json

class Dataset_tu(BaseDataset):
    CLASSES = ['hair']
    def __init__(
            self, 
            mode = 'train',
            classes=None, 
            augmentation=None, 
            preprocessing=None,
    ):
        self.data_dir = 'C:\\hairdata'

        if mode == 'train':
            path = os.path.join(self.data_dir, 'task02_train')
        elif mode =='val':
            path = os.path.join(self.data_dir, 'val')
        else:
            print("testtttttttttt")
            path = os.path.join(self.data_dir, 'task02_test')


        #if not os.path.isdir(os.path.join(path, 'masks')):
        #    os.makedirs(path+'masks')
        self.labels_fps = os.path.join(path, 'labels')
        self.labels_fps = [os.path.join(path, i) for i in self.labels_fps]
        #self.polygon_to_mask(path)

        self.images_fps = os.listdir(os.path.join(path, 'images'))
        self.images_fps = [os.path.join(path, 'images', i) for i in self.images_fps]
        self.file_name = os.listdir(os.path.join(path+'/', 'masks'))
        self.masks_fps = [os.path.join(path, 'masks', i) for i in self.file_name]

        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #cv2.waitKey(0)

        mask = cv2.imread(self.masks_fps[i],0)
    
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

    # def polygon_to_mask(self, path):
    #     mask_dir = os.path.join(path,'masks')
    #     json_file = utils.load_json(os.path.join(path,'labels.json'))

    #     json_file['annotations'] = sorted(json_file['annotations'], key = lambda x: list(x.items())[0])

    #     for i, file_path in enumerate(os.listdir(os.path.join(path,'images'))):
    #         polygon = []
    #         for line in json_file['annotations'][i]['polygon1']:
    #             xy = list(line.values())
    #             polygon.append((xy[0], xy[1]))

    #         img = Image.new('L', (512, 512), 'black')
    #         ImageDraw.Draw(img).polygon(polygon, outline='white', fill='white')
    #         img.save(os.path.join(mask_dir,file_path.split('.')[0]+'.jpg'))
import random

def rand_bbox(size, lam):
    if len(size) == 4:
        W = size[2]
        H = size[3]
    elif len(size) == 3:
        W = size[1]
        H = size[2]
    else:
        raise Exception

    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

class CutMix(Dataset_tu):
    def __init__(self, datasett, num_mix=1, beta=1., prob = 0.8):
        self.datasett = datasett
        self.num_mix = num_mix
        self.beta = beta
        self.prob = prob

    def __getitem__(self, i):
        imaage, masssk = self.datasett[i]
        for _ in range(self.num_mix):
            r = np.random.rand(1)
            if self.beta <= 0 or r > self.prob:
                continue

            lam = np.random.beta(self.beta, self.beta)
            rand_i = random.choice(range(len(self)))

            imaage2, masssk2 = self.datasett[rand_i]

            bbx1, bby1, bbx2, bby2 = rand_bbox((3,512,512), lam)
            imaage[:, bbx1:bbx2, bby1:bby2] = imaage2[:, bbx1:bbx2, bby1:bby2]
            masssk[:, bbx1:bbx2, bby1:bby2] = masssk2[:, bbx1:bbx2, bby1:bby2]

        return imaage, masssk

    def __len__(self):
        return len(self.datasett)

class Dataset_pred(BaseDataset):
    CLASSES = ['hair']
    def __init__(
            self, 
            mode = 'test',
            classes=None, 
            augmentation=None, 
            preprocessing=None,
    ):
        self.data_dir = 'C:\\hairdata'

        if mode == 'train':
            path = os.path.join(self.data_dir, 'task02_train')
        elif mode =='val':
            path = os.path.join(self.data_dir, 'val')
        else:
            print("testtttttttttt")
            path = os.path.join(self.data_dir, 'task02_test')


        #if not os.path.isdir(os.path.join(path, 'masks')):
        #    os.makedirs(path+'masks')
        self.labels_fps = os.path.join(path, 'labels')
        self.labels_fps = [os.path.join(path, i) for i in self.labels_fps]
        # self.polygon_to_mask(path)

        self.images_fps = os.listdir(os.path.join(path, 'images'))
        self.images_fps = [os.path.join(path, 'images', i) for i in self.images_fps]
        self.file_name = os.listdir(os.path.join(path+'/', 'masks'))
        self.masks_fps = [os.path.join(path, 'masks', i) for i in self.file_name]

        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
            
        # read data
        
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #image = image.transpose(2,0,1).astype('float32')
        # cv2.imshow("img",image)
        # cv2.waitKey(0)

        # mask = cv2.imread(self.masks_fps[i],0)
        mask = cv2.imread(self.images_fps[i])
    
        # masks = [(mask == 255)]
        # mask = np.stack(masks, axis=-1).astype('float')

        # # apply augmentations
        # if self.augmentation:
        #     sample = self.augmentation(image=image, mask=mask)
        #     image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        return image ,mask
        
    def __len__(self):
        return len(self.images_fps)

    def polygon_to_mask(self, path):
        '''
        #os.makedirs(os.path.join(path,'masks'))
        mask_dir = os.path.join(path,'masks3')
        print(path)
        #os.makedirs(os.path.join(path,'labels'))
        label_dir = os.path.join(path,'labels')
        i=0
        for file_path in os.listdir(os.path.join(path,'images')):
            
            # label_file = open(os.path.join(path+'/','labels/'))
            # print(label_file)
            
            # polygon = []
            # for line in label_file:
            #     x,y = line.split(' ')
            #     x,y = float(x), float(y)
            #     polygon.append((x,y))
            json_file = utils.load_json(os.path.join(path,'labels.json'))

        
            print(json_file)
            polygon_dict = json_file['annotations'][i]['polygon1']

            file_name = json_file['annotations'][i]['file_name']
            # if type(polygon_dict) is dict:
            #         polygons = [r['annotations'] for r in ['polygon1'].values()]
            # else:
            #     polygons = [r['annotations'] for r in ['polygon1']]

            #print(polygons)
            #print(polygon_dict)
            i+=1
            #polygon_dict = polygon_dict['polygon1']
            #annotations = json_file['annotations']
            #for a in annotations:
            #    polygon_dict = a['polygon1']
            polygon = []
            
            for point in polygon_dict:
                polygon.append(tuple(point.values()))
            
            img = Image.new('L', (512, 512), 'black')
            ImageDraw.Draw(img).polygon(polygon, outline='white', fill='white')
            img.save(os.path.join(mask_dir,file_name + '.jpg'))
            
            '''

if __name__=='__main__':
    a = Dataset_tu(mode= 'train')