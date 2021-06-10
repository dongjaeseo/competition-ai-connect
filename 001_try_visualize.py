import cv2
import json
import ast
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageDraw

for i in range(10):
    img = Image.open(f'./sample_hairstyle/sample_image/img{i+1:02d}.jpg').convert('RGBA')
    # plt.imshow(img)
    # plt.show()

    with open('./sample_hairstyle/sample_label.json') as f:
        label = json.load(f)

    coordinates = label['annotation'][i]['polygon1']
    coordinates = ast.literal_eval(coordinates)

    x = []
    y = []
    for coor in coordinates:
        x.append(coor['x'])
        y.append(coor['y'])

    x = map(float, x)
    y = map(float, y)
    xy = list(zip(x, y))

    img2 = img.copy()
    draw = ImageDraw.Draw(img2)
    draw.polygon(xy, fill = 'wheat')

    img3 = Image.blend(img, img2, 0.5)
    plt.imshow(img3)
    plt.show()