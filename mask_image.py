import numpy
import os
import cv2

'''
list = os.listdir('C:/Users/Choi Jun Ho/ai_competition_hairstyle/train/masks2/')
a = []
for i in list:
    sum = os.path.splitext(i)
    a.append(sum[0])


for i in a:
    rgb_image = cv2.imread('C:/Users/Choi Jun Ho/ai_competition_hairstyle/train2/images/%s'%i)
    cv2.imwrite('C:/Users/Choi Jun Ho/ai_competition_hairstyle/new_train/%s'%i,rgb_image)
'''

list = os.listdir('C:/Users/Choi Jun Ho/ai_competition_hairstyle/train/masks2/')
a=[]
for i in list[50:]:
    sum = os.path.splitext(i)
    a.append(sum[0])
    

for i in a:
    rgb_image = cv2.imread('C:/Users/Choi Jun Ho/ai_competition_hairstyle/train2/images/%s'%i)
    cv2.imwrite('C:/Users/Choi Jun Ho/ai_competition_hairstyle/new_train3/%s'%i,rgb_image)