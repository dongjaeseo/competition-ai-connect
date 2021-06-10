import cv2
import json
import ast
import matplotlib.pyplot as plt

# img = cv2.imread('./sample_hairstyle/sample_image/img01.jpg')
# cv2.imshow('img', img)
# cv2.waitKey(0)

with open('./sample_hairstyle/sample_label.json') as f:
    label = json.load(f)

coordinates = label['annotation'][0]['polygon1']
coordinates = ast.literal_eval(coordinates)

for coor in coordinates:
    x = coor['x']
    y = coor['y']
    print(x,y)