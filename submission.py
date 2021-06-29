import torch
import numpy as np
from tqdm import tqdm
import segmentation_models_pytorch as smp
import argparse
import os
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils import data as data_utils
import albumentations as albu
from torchsummary import summary
from efficientnet_pytorch import EfficientNet
import torchvision.models as models
import torch.onnx as onnx
from datetime import datetime, timezone, timedelta
import random
import logging
# from modules.dataset import CustomDataset

import cv2
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')
print('Using pytorch version:',torch.__version__,'Device:',DEVICE) #Using pytorch version: 1.7.1 Device: cuda

class TestDataset(data_utils.Dataset):
    """__init__ and __len__ functions are the same as in TorchvisionDataset"""
    def __init__(self, data_dir, imgs):
        self.data_dir = data_dir
        self.imgs = imgs

    def __getitem__(self, idx):
        filename = self.imgs[idx]
        # Read an image with OpenCV
        img = cv2.imread(os.path.join(self.data_dir, self.imgs[idx]))
        print(filename,img)
        return filename, img

logger = logging.getLogger(__name__)
final_output_dir= 'C:/Users/Choi Jun Ho/ai_competition_hairstyle/baseline/results/train/20210624152622/'
model = models.resnet101(pretrained= False).cuda()
model_state_file = os.path.join(final_output_dir, 'best.pt')
logger.info('=> loading model from {}'.format(model_state_file))
model.load_state_dict(torch.load(model_state_file))
model = model.eval()
#model = torch.nn.DataParallel(model).cuda()
logger.info("=> Model Creation Success")

test_dir = f'C:/Users/Choi Jun Ho/ai_competition_hairstyle/test/images/'
test_imgs = os.listdir(test_dir)
test_data = TestDataset(test_dir, test_imgs)

'''
PATH = 'C:/Users/Choi Jun Ho/ai_competition_hairstyle/baseline/results/train/20210622181558/last.pt'
#test_path = test_dir = f'C:/Users/Choi Jun Ho/ai_competition_hairstyle/test/images/image_00004369548859.jpg'
model = torch.load(PATH)
#image = torch.load(test_path)
#predicted = model.predict(image)
input_image = f'C:/Users/Choi Jun Ho/ai_competition_hairstyle/test/images/image_00004369548859.jpg' 

test_dir = f'C:/Users/Choi Jun Ho/ai_competition_hairstyle/test/images/'
test_imgs = os.listdir(test_dir)
test_data = TestDataset(test_dir, test_imgs)
test_loader = data_utils.DataLoader(test_data, batch_size=8, shuffle=False)

all_predictions = []
files = []
with torch.no_grad():
    for filenames, inputs in test_loader:
        print(filenames)
        print(inputs)
        # image = image.to(DEVICE)
        # label = label.to(DEVICE)
        # output = model(image)
        # predictions = list(model(inputs.to(DEVICE)).cpu().numpy())
        # files.extend(filenames)
        # for prediction in predictions:
        #     all_predictions.append(prediction)
        #     print(all_predictions)

'''