from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset

import os
import numpy as np
import cv2
import pandas as pd
from PIL import Image, ImageDraw
import modules.utils as utils
import json

path = "C:/Users/Choi Jun Ho/ai_competition_hairstyle/test/"
mask_dir = os.path.join(path,'masks4')
json_file = utils.load_json(os.path.join(path,'submit3.json'))

json_file['annotations'] = sorted(json_file['annotations'], key = lambda x: list(x.items())[0])

for i, file_path in enumerate(os.listdir(os.path.join(path,'images'))):
    polygon = []
    for line in json_file['annotations'][i]['polygon1']:
        xy = list(line.values())
        polygon.append((xy[0], xy[1]))

    img = Image.new('L', (512, 512), 'black')
    ImageDraw.Draw(img).polygon(polygon, outline='white', fill='white')
    img.save(os.path.join(mask_dir,file_path.split('.')[0]+'.jpg'))
            