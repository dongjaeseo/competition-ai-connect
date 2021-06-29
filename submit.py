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
    #contours2 = contours2.reshape((contours2[0]*contours2[1]),contours2[2],contours2[3])
    #print(contours2.shape)
    
    '''
    contours3=[]
    for x in contours2:
        for number,y in enumerate(x):
            for z in y:
                contours3.append(contours2[0][number])

    contours3 = np.array(contours2)
    print(contours3.shape)
    '''
    '''
    cv2.drawContours(img, contours2, 0, (0, 255, 0), 3) # 인덱스0, 파란색
    #cv2.drawContours(img, contours, 1, (255, 0, 0), 3) # 인덱스1, 초록색
    # cv2.imshow("result", img)
    # cv2.waitKey(0)

    for x in contours2:
        for number,y in enumerate(x):
            for z in y:
                if len(x) < 20:
                    json_file['annotations'][i]['polygon1'].append(dict(x=int(z[0]),y=int(z[1])))
                elif 20<= len(x) and 40>len(x):
                    if number%2==0:
                        json_file['annotations'][i]['polygon1'].append(dict(x=int(z[0]),y=int(z[1])))
                elif 40<= len(x) and 70>len(x):
                    if number%4==0:
                        json_file['annotations'][i]['polygon1'].append(dict(x=int(z[0]),y=int(z[1])))
                elif 70<= len(x) and 120>len(x):
                    if number%6==0:
                        json_file['annotations'][i]['polygon1'].append(dict(x=int(z[0]),y=int(z[1])))
                elif 120<= len(x) and 180>len(x):
                    if number%8==0:
                        json_file['annotations'][i]['polygon1'].append(dict(x=int(z[0]),y=int(z[1])))
                elif 180<= len(x) and 280>len(x):
                    if number%8==0:
                        json_file['annotations'][i]['polygon1'].append(dict(x=int(z[0]),y=int(z[1])))
                elif 280<= len(x) and 400>len(x):
                    if number%10==0:
                        json_file['annotations'][i]['polygon1'].append(dict(x=int(z[0]),y=int(z[1])))
                elif 400<= len(x):
                    if number%15==0:
                        json_file['annotations'][i]['polygon1'].append(dict(x=int(z[0]),y=int(z[1])))

utils.save_json('C:/Users/Choi Jun Ho/ai_competition_hairstyle/test/submit3.json', json_file)
'''
    #cv2.drawContours(img, contours2, 0, (0, 255, 0), 3) # 인덱스0, 파란색
    #cv2.drawContours(img, contours, 1, (255, 0, 0), 3) # 인덱스1, 초록색
    # cv2.imshow("result", img)
    # cv2.waitKey(0)

    for x in contours2:
        for number,y in enumerate(x):
            for z in y:
                if number%2==0:
                    json_file['annotations'][i]['polygon1'].append(dict(x=int(z[0]),y=int(z[1])))

#json_file = str(json_file)
#json_file = json_file.replace(' ', '')
#print(json_file)
utils.save_json(os.path.join(test_mask_dir,'submit5.json') , json_file)