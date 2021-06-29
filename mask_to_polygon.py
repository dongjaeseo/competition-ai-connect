import numpy as np
from imantics import Polygons, Mask


from PIL import Image
test_img = "C:/Users/Choi Jun Ho/ai_competition_hairstyle/train/masks3/image_00019788408186.jpg.jpg"

img = Image.open(test_img)
array_img = np.array(img, dtype='uint8')


polygons = Mask(array_img).polygons()

print(polygons.points)
print(polygons.segmentation)