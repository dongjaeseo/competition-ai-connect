import torch
import numpy as np
from tqdm import tqdm
import segmentation_models_pytorch as smp
import argparse
import os
import torch.optim as optim
from torch.utils.data import DataLoader
from modules.dataset import Dataset_tu, Dataset_pred
import albumentations as albu
#import tensorflow as tf
from modules.utils import load_yaml, save_yaml, get_logger, make_directory
import torchsummary
from datetime import datetime, timezone, timedelta
import random
#from models.hrnet import HRNet 
# from modules.dataset import CustomDataset
from modules.earlystoppers import LossEarlyStopper
from modules.recorders import PerformanceRecorder
from modules.trainer import CustomTrainer
from modules.utils import load_yaml, save_yaml, get_logger, make_directory
from efficientnet_pytorch import EfficientNet
from torch.utils import data as data_utils
import cv2
import torch.nn as nn
import torchvision.models as models
import torch.onnx as onnx
from datetime import datetime, timezone, timedelta
import random
import logging

import torch
from torch import nn
#from models.modules import BasicBlock, Bottleneck
import torch
from torch import nn


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, bn_momentum=0.1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=bn_momentum)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=bn_momentum)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, bn_momentum=0.1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=bn_momentum)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class StageModule(nn.Module):
    def __init__(self, stage, output_branches, c, bn_momentum):
        super(StageModule, self).__init__()
        self.stage = stage
        self.output_branches = output_branches

        self.branches = nn.ModuleList()
        for i in range(self.stage):
            w = c * (2 ** i)
            branch = nn.Sequential(
                BasicBlock(w, w, bn_momentum=bn_momentum),
                BasicBlock(w, w, bn_momentum=bn_momentum),
                BasicBlock(w, w, bn_momentum=bn_momentum),
                BasicBlock(w, w, bn_momentum=bn_momentum),
            )
            self.branches.append(branch)

        self.fuse_layers = nn.ModuleList()
        # for each output_branches (i.e. each branch in all cases but the very last one)
        for i in range(self.output_branches):
            self.fuse_layers.append(nn.ModuleList())
            for j in range(self.stage):  # for each branch
                if i == j:
                    self.fuse_layers[-1].append(nn.Sequential())  # Used in place of "None" because it is callable
                elif i < j:
                    self.fuse_layers[-1].append(nn.Sequential(
                        nn.Conv2d(c * (2 ** j), c * (2 ** i), kernel_size=(1, 1), stride=(1, 1), bias=False),
                        nn.BatchNorm2d(c * (2 ** i), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                        nn.Upsample(scale_factor=(2.0 ** (j - i)), mode='nearest'),
                    ))
                elif i > j:
                    ops = []
                    for k in range(i - j - 1):
                        ops.append(nn.Sequential(
                            nn.Conv2d(c * (2 ** j), c * (2 ** j), kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),
                                      bias=False),
                            nn.BatchNorm2d(c * (2 ** j), eps=1e-05, momentum=0.1, affine=True,
                                           track_running_stats=True),
                            nn.ReLU(inplace=True),
                        ))
                    ops.append(nn.Sequential(
                        nn.Conv2d(c * (2 ** j), c * (2 ** i), kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),
                                  bias=False),
                        nn.BatchNorm2d(c * (2 ** i), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    ))
                    self.fuse_layers[-1].append(nn.Sequential(*ops))

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        assert len(self.branches) == len(x)

        x = [branch(b) for branch, b in zip(self.branches, x)]

        x_fused = []
        for i in range(len(self.fuse_layers)):
            for j in range(0, len(self.branches)):
                if j == 0:
                    x_fused.append(self.fuse_layers[i][0](x[0]))
                else:
                    x_fused[i] = x_fused[i] + self.fuse_layers[i][j](x[j])

        for i in range(len(x_fused)):
            x_fused[i] = self.relu(x_fused[i])

        return x_fused


class HRNet(nn.Module):
    def __init__(self, c=48, nof_joints=17, bn_momentum=0.1):
        super(HRNet, self).__init__()

        # Input (stem net)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=bn_momentum, affine=True, track_running_stats=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(64, eps=1e-05, momentum=bn_momentum, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)

        # Stage 1 (layer1)      - First group of bottleneck (resnet) modules
        downsample = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=bn_momentum, affine=True, track_running_stats=True),
        )
        self.layer1 = nn.Sequential(
            Bottleneck(64, 64, downsample=downsample),
            Bottleneck(256, 64),
            Bottleneck(256, 64),
            Bottleneck(256, 64),
        )

        # Fusion layer 1 (transition1)      - Creation of the first two branches (one full and one half resolution)
        self.transition1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256, c, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                nn.BatchNorm2d(c, eps=1e-05, momentum=bn_momentum, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(nn.Sequential(  # Double Sequential to fit with official pretrained weights
                nn.Conv2d(256, c * (2 ** 1), kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
                nn.BatchNorm2d(c * (2 ** 1), eps=1e-05, momentum=bn_momentum, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
            )),
        ])

        # Stage 2 (stage2)      - Second module with 1 group of bottleneck (resnet) modules. This has 2 branches
        self.stage2 = nn.Sequential(
            StageModule(stage=2, output_branches=2, c=c, bn_momentum=bn_momentum),
        )

        # Fusion layer 2 (transition2)      - Creation of the third branch (1/4 resolution)
        self.transition2 = nn.ModuleList([
            nn.Sequential(),  # None,   - Used in place of "None" because it is callable
            nn.Sequential(),  # None,   - Used in place of "None" because it is callable
            nn.Sequential(nn.Sequential(  # Double Sequential to fit with official pretrained weights
                nn.Conv2d(c * (2 ** 1), c * (2 ** 2), kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
                nn.BatchNorm2d(c * (2 ** 2), eps=1e-05, momentum=bn_momentum, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
            )),  # ToDo Why the new branch derives from the "upper" branch only?
        ])

        # Stage 3 (stage3)      - Third module with 4 groups of bottleneck (resnet) modules. This has 3 branches
        self.stage3 = nn.Sequential(
            StageModule(stage=3, output_branches=3, c=c, bn_momentum=bn_momentum),
            StageModule(stage=3, output_branches=3, c=c, bn_momentum=bn_momentum),
            StageModule(stage=3, output_branches=3, c=c, bn_momentum=bn_momentum),
            StageModule(stage=3, output_branches=3, c=c, bn_momentum=bn_momentum),
        )

        # Fusion layer 3 (transition3)      - Creation of the fourth branch (1/8 resolution)
        self.transition3 = nn.ModuleList([
            nn.Sequential(),  # None,   - Used in place of "None" because it is callable
            nn.Sequential(),  # None,   - Used in place of "None" because it is callable
            nn.Sequential(),  # None,   - Used in place of "None" because it is callable
            nn.Sequential(nn.Sequential(  # Double Sequential to fit with official pretrained weights
                nn.Conv2d(c * (2 ** 2), c * (2 ** 3), kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
                nn.BatchNorm2d(c * (2 ** 3), eps=1e-05, momentum=bn_momentum, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
            )),  # ToDo Why the new branch derives from the "upper" branch only?
        ])

        # Stage 4 (stage4)      - Fourth module with 3 groups of bottleneck (resnet) modules. This has 4 branches
        self.stage4 = nn.Sequential(
            StageModule(stage=4, output_branches=4, c=c, bn_momentum=bn_momentum),
            StageModule(stage=4, output_branches=4, c=c, bn_momentum=bn_momentum),
            StageModule(stage=4, output_branches=1, c=c, bn_momentum=bn_momentum),
        )

        # Final layer (final_layer)
        self.final_layer = nn.Conv2d(c, nof_joints, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = [trans(x) for trans in self.transition1]  # Since now, x is a list (# == nof branches)

        x = self.stage2(x)
        # x = [trans(x[-1]) for trans in self.transition2]    # New branch derives from the "upper" branch only
        x = [
            self.transition2[0](x[0]),
            self.transition2[1](x[1]),
            self.transition2[2](x[-1])
        ]  # New branch derives from the "upper" branch only

        x = self.stage3(x)
        # x = [trans(x) for trans in self.transition3]    # New branch derives from the "upper" branch only
        x = [
            self.transition3[0](x[0]),
            self.transition3[1](x[1]),
            self.transition3[2](x[2]),
            self.transition3[3](x[-1])
        ]  # New branch derives from the "upper" branch only

        x = self.stage4(x)

        x = self.final_layer(x[0])

        return x


if __name__ == '__main__':
    # model = HRNet(48, 17, 0.1)
    model = HRNet(32, 17, 0.1)

    # print(model)

    model.load_state_dict(
        # torch.load('./weights/pose_hrnet_w48_384x288.pth')
        torch.load('./weights/pose_hrnet_w32_256x192.pth')
    )
    print('ok!!')

    if torch.cuda.is_available() and False:
        torch.backends.cudnn.deterministic = True
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    print(device)

    model = model.to(device)

    y = model(torch.ones(1, 3, 384, 288).to(device))
    print(y.shape)
    print(torch.min(y).item(), torch.mean(y).item(), torch.max(y).item())

iou_thres = 0.75

# DEBUG
DEBUG = False

# CONFIG
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_PROJECT_DIR = os.path.dirname(PROJECT_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
TRAIN_CONFIG_PATH = os.path.join(PROJECT_DIR, 'config/train_config.yaml')
config = load_yaml(TRAIN_CONFIG_PATH)
# print(PROJECT_DIR)
# print(ROOT_PROJECT_DIR)
# print(DATA_DIR)


# SEED
RANDOM_SEED = config['SEED']['random_seed']

# DATALOADER
NUM_WORKERS = config['DATALOADER']['num_workers']
CHECKPOINT_PATH = ''
PIN_MEMORY = config['DATALOADER']['pin_memory']

# TRAIN
EPOCHS = config['TRAIN']['num_epochs']
TRAIN_BATCH_SIZE = config['TRAIN']['batch_size']
MODEL = config['TRAIN']['model']
LEARNING_RATE = config['TRAIN']['learning_rate']
EARLY_STOPPING_PATIENCE = config['TRAIN']['early_stopping_patience']
OPTIMIZER = config['TRAIN']['optimizer']
SCHEDULER = config['TRAIN']['scheduler']
MOMENTUM = config['TRAIN']['momentum']
WEIGHT_DECAY = config['TRAIN']['weight_decay']
LOSS_FN = config['TRAIN']['loss_function']
METRIC_FN = config['TRAIN']['metric_function']
# VALIDATION
EVAL_BATCH_SIZE = config['VALIDATION']['batch_size']

# TRAIN SERIAL
KST = timezone(timedelta(hours=9))
TRAIN_TIMESTAMP = datetime.now(tz=KST).strftime("%Y%m%d%H%M%S")
TRAIN_SERIAL = f'{TRAIN_TIMESTAMP}' if DEBUG is not True else 'DEBUG'

# PERFORMANCE RECORD
PERFORMANCE_RECORD_DIR = os.path.join(PROJECT_DIR, 'results', 'train', TRAIN_SERIAL)
PERFORMANCE_RECORD_COLUMN_NAME_LIST = config['PERFORMANCE_RECORD']['column_list']
'''
class MaskTestDataset(Dataset):
    
    def __init__(self, root_dir, transforms=None, mode='test'):
        """
        Args:
            root_dir   : Path of Dataset
            box_dir    : Box coordinate File 
            transforms : Augmentation(Flip, Rotate ...)
        """

        self.root_dir = root_dir
        self.transforms = transforms
        self.mode = mode
        self.img_list = [file for file in os.listdir(os.path.join(self.root_dir,self.mode)) if file.endswith(".jpg")]

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_id = self.img_list[idx]

        img_path = os.path.join(self.root_dir,self.mode,img_id)            
        img = cv2.imread(img_path)

        if self.transforms:
            augmented = self.transforms(image=img)
            img    = augmented['image']

        return img_id, img
'''
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

if __name__ == '__main__':
    torch.manual_seed(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    
    ENCODER = 'resnet10111111'
    ENCODER_WEIGHTS = 'imagenet'
    CLASSES = ['hair']
    ACTIVATION = 'sigmoid'
    

    import torch

    USE_CUDA = torch.cuda.is_available()
    print(USE_CUDA)

    device = torch.device('cuda:0' if USE_CUDA else 'cpu')
    print('학습을 진행하는 기기:',device)

    print('cuda index:', torch.cuda.current_device())

    print('gpu 개수:', torch.cuda.device_count())

    print('graphic name:', torch.cuda.get_device_name())
    DEVICE = torch.device('cuda')

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


    # Set train result directory
    make_directory(PERFORMANCE_RECORD_DIR)

    # Set system logger
    system_logger = get_logger(name='train', 
                               file_path=os.path.join(PERFORMANCE_RECORD_DIR, 'train_log.log'))
    
    # Unet / PSPNet / DeepLabV3Plus
    if MODEL == 'unet':
        print("uuuuuu")
        # model = models.resnet101(pretrained= False).cuda()
        # num_ftrs = model.fc.in_features
        # model.fc = nn.Linear(num_ftrs,2)
        # model = model.cuda()
        
        model = smp.Unet(
            encoder_name=ENCODER, 
            encoder_weights=ENCODER_WEIGHTS, 
            classes=len(CLASSES), 
            activation=ACTIVATION,
        )
        torchsummary.summary(model, (3,512,512),device='cpu')
    elif MODEL == 'pspnet':
        print("psp")

        model = smp.PSPNet(
            encoder_name=ENCODER, 
            encoder_weights=ENCODER_WEIGHTS, 
            classes=len(CLASSES), 
            activation=ACTIVATION,
        )
    elif MODEL == 'deeplabv3plus':
        print("deep")
        model = smp.DeepLabV3Plus(
            encoder_name=ENCODER, 
            encoder_weights=ENCODER_WEIGHTS, 
            classes=len(CLASSES), 
            activation=ACTIVATION,
        )
    elif  MODEL == 'pannet':
        print("pannet")
        model = smp.PAN(
            encoder_name=ENCODER,
            encoder_weights=ENCODER_WEIGHTS,
            classes=len(CLASSES),
            activation=ACTIVATION,
        )
    elif  MODEL == 'fpn':
        print("fpn")
        model = smp.FPN(
            encoder_name=ENCODER,
            encoder_weights=ENCODER_WEIGHTS,
            classes=len(CLASSES),
            activation=ACTIVATION,
        )
    elif MODEL == 'hrnet':
        model = HRNet(c=48, nof_joints=17,
                           bn_momentum=0.1).to(DEVICE)
    else:
        raise RuntimeError('Model name Error')

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    # ------------------------------------------------------
    # 데이터 로드

    def to_tensor(x, **kwargs):
        return x.transpose(2, 0, 1).astype('float32')


    def get_preprocessing(preprocessing_fn):
   
        _transform = [
            albu.Lambda(image=preprocessing_fn),
            albu.Lambda(image=to_tensor, mask=to_tensor),
        ]
      
        return albu.Compose(_transform)


    train_dataset = Dataset_tu(
        mode = 'train',
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=CLASSES,
    )

    # val_dataset = Dataset_tu(
    #     mode = 'train',
    #     preprocessing=get_preprocessing(preprocessing_fn),
    #     classes=CLASSES,
    # )
    
    test_dataset = Dataset_pred(
        mode = 'test',
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=CLASSES,
    )

    # Load dataset & dataloader
    train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, num_workers=0, pin_memory=PIN_MEMORY, shuffle=True, drop_last=True)
    #valid_loader = DataLoader(val_dataset, batch_size=EVAL_BATCH_SIZE, num_workers=0, pin_memory=PIN_MEMORY, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=EVAL_BATCH_SIZE, num_workers=0, pin_memory=PIN_MEMORY, shuffle=False, drop_last=False)
    print(len(train_loader))
    #print(len(valid_loader))
    #print(len(test_loader))


    #image_88036057016182.jpg
    #image_64317716451325.jpg
    # -------------------------------------------------------------
    # 모델 학습
    #9시 10분 시작 
    loss = smp.utils.losses.DiceLoss()
    
    # IOU / Fscore / Accuracy
    metrics = [
        smp.utils.metrics.IoU(threshold=iou_thres),
        smp.utils.metrics.Fscore(threshold=iou_thres),
        smp.utils.metrics.Accuracy(threshold=iou_thres),
    ]

    optimizer = torch.optim.Adam([ 
        dict(params=model.parameters(), lr=LEARNING_RATE),
    ])
    if CHECKPOINT_PATH != '':
        save_path = os.path.join(CHECKPOINT_PATH, MODEL)
        model = torch.load(os.path.join(save_path, f'best_model.pth'))
        print('load complete')
    else:
        save_path = './checkpoints/' + MODEL
        if os.path.exists(save_path) == False:
            os.makedirs(save_path)


    train_epoch = smp.utils.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )

    # valid_epoch = smp.utils.train.ValidEpoch(
    #     model,
    #     loss=loss,
    #     metrics=metrics,
    #     device=DEVICE,
    #     verbose=True,
    # )

    # predict_epoch = smp.utils.train.PredictEpoch(
    #     model,
    #     loss=loss,
    #     metrics=metrics,
    #     device=DEVICE,
    #     verbose=True,
    # )

    
    # Load Model
    system_logger.info('===== Review Model Architecture =====')
    system_logger.info(f'{model} \n')

    # Set optimizer, scheduler, loss function, metric function
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=1e5, max_lr=0.0001, epochs=EPOCHS, steps_per_epoch=len(train_loader))
   
    #metric_fn = mean_squared_error

    # Set trainer

    trainer = CustomTrainer(model, DEVICE, loss, metrics, optimizer, scheduler, logger=system_logger)

    # Set earlystopper
    early_stopper = LossEarlyStopper(patience=EARLY_STOPPING_PATIENCE, verbose=True, logger=system_logger)

    # Set performance recorder
    key_column_value_list = [
        TRAIN_SERIAL,
        TRAIN_TIMESTAMP,
        MODEL,
        OPTIMIZER,
        LOSS_FN,
        METRIC_FN,
        EARLY_STOPPING_PATIENCE,
        TRAIN_BATCH_SIZE,
        EPOCHS,
        LEARNING_RATE,
        WEIGHT_DECAY,
        RANDOM_SEED]

    performance_recorder = PerformanceRecorder(column_name_list=PERFORMANCE_RECORD_COLUMN_NAME_LIST,
                                               record_dir=PERFORMANCE_RECORD_DIR,
                                               key_column_value_list=key_column_value_list,
                                               logger=system_logger,
                                               model=model,
                                               optimizer=optimizer,
                                               scheduler=scheduler)

    # Save config yaml file
    save_yaml(os.path.join(PERFORMANCE_RECORD_DIR, 'train_config.yaml'), config)
    trainer.predict_epoch(test_loader, epoch_index=0, verbose=False, logging_interval=1)
    '''
    # Train
    for epoch_index in tqdm(range(EPOCHS)):

        train_logs = train_epoch.run(train_loader)
        #valid_logs = valid_epoch.run(valid_loader)
        print('epoch : ',epoch_index)
        print('train_logs', train_logs)
        #print('val_logs', valid_logs)

        #val_loss = valid_logs['dice_loss']

        
        # Performance record - csv & save elapsed_time
        performance_recorder.add_row(epoch_index=epoch_index,
                                     train_loss=trainer.train_loss_mean,
                                     validation_loss=trainer.validation_loss_mean,
                                     train_score=trainer.train_score,
                                     validation_score=trainer.validation_score)

        # Performance record - plot
        performance_recorder.save_performance_plot(final_epoch=epoch_index)

        # early_stopping check
        early_stopper.check_early_stopping(loss=trainer.validation_loss_mean)
        
        if early_stopper.stop:
            break

        trainer.clear_history()
    
    trainer.predict_epoch(test_loader, epoch_index=0, verbose=False, logging_interval=1)
    
    
    final_output_dir= 'C:/Users/Choi Jun Ho/ai_competition_hairstyle/baseline/results/train/20210627233220/'
    # last model save
    performance_recorder.weight_path = os.path.join(PERFORMANCE_RECORD_DIR, 'last.pt')
    performance_recorder.save_weight()
    '''

    # logger = logging.getLogger(__name__)
    # final_output_dir= 'C:/Users/Choi Jun Ho/ai_competition_hairstyle/baseline/results/train/20210627233220/best.pt'
    # model = torch.load(final_output_dir)
    # model_state_file = os.path.join(final_output_dir, 'best.pt')
    # logger.info('=> loading model from {}'.format(model_state_file))
    # model.load_state_dict(torch.load(model_state_file))
    # model = model.eval()
    # #model = torch.nn.DataParallel(model).cuda()
    # logger.info("=> Model Creation Success")

    # test_dir = f'C:/Users/Choi Jun Ho/ai_competition_hairstyle/test/images/'
    # test_imgs = os.listdir(test_dir)
    # test_data = TestDataset(test_dir, test_imgs)
        # logger = logging.getLogger(__name__)
        # final_output_dir= 'C:/Users/Choi Jun Ho/ai_competition_hairstyle/baseline/results/train/20210624152622/last.pt'
        # model = smp.Unet(
        #         encoder_name=ENCODER, 
        #         encoder_weights=ENCODER_WEIGHTS, 
        #         classes=len(CLASSES), 
        #         activation=ACTIVATION,
        #     )
        # model_state_file = os.path.join(final_output_dir, 'best.pt')
        # logger.info('=> loading model from {}'.format(model_state_file))
        # model.load_state_dict(torch.load(model_state_file))
        # model = model.eval()
        # #model = torch.nn.DataParallel(model).cuda()
        # logger.info("=> Model Creation Success")

        # test_dir = f'C:/Users/Choi Jun Ho/ai_competition_hairstyle/test/images/'
        # test_imgs = os.listdir(test_dir)
        # test_data = TestDataset(test_dir, test_imgs)


    ######################################    
    # final_output_dir= 'C:/Users/Choi Jun Ho/ai_competition_hairstyle/baseline/results/train/20210627233220/best.pt'
    # model = torch.load(final_output_dir)

    # model.eval()
    
    # with torch.no_grad():
    #     for image, label in test_loader:
    #         image = image.to(DEVICE)
    #         label = label.to(DEVICE)
    #         output = model(image)