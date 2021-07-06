import json
import os
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
from imantics import Polygons, Mask
from modules import utils
from tqdm import tqdm
import cv2
import ast

test_mask_dir = 'C:\\hairdata\\task02_test\\'
json_file = utils.load_json(os.path.join(test_mask_dir,'sample_submission.json'))

for i, file_path in tqdm(enumerate(os.listdir(os.path.join(test_mask_dir,'masks')))):

    img = cv2.imread(os.path.join(test_mask_dir, 'masks', file_path),cv2.IMREAD_GRAYSCALE)
    
    ret, img_binary = cv2.threshold(img, 127, 255, 0)

    contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) # 컨투어 검출
    contours2 = np.array(contours)

    for x in contours2:
        for number,y in enumerate(x):
            for z in y:
                if number %2 == 0 :
                    json_file['annotations'][i]['polygon1'].append(dict(x=int(z[0]),y=int(z[1])))

#json_file = str(json_file)
#json_file = json_file.replace(' ', '')
#print(json_file)
utils.save_json(os.path.join(test_mask_dir,'submit0702_1200.json') , json_file)