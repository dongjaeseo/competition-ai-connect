import torch
import numpy as np
from tqdm import tqdm
import segmentation_models_pytorch as smp
import argparse
import os
import torch.optim as optim
from torch.utils.data import DataLoader
from modules.dataset import Dataset_tu, Dataset_pred, CutMix
import albumentations as albu
#import tensorflow as tf
from modules.utils import load_yaml, save_yaml, get_logger, make_directory
import torchsummary
from datetime import datetime, timezone, timedelta
import random

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
    
    ENCODER = 'resnet101'
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
        classes=CLASSES
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

    import matplotlib.pyplot as plt
    # Load dataset & dataloader
    train_loader = DataLoader(CutMix(train_dataset), batch_size=TRAIN_BATCH_SIZE, num_workers=0, pin_memory=PIN_MEMORY, shuffle=True, drop_last=True)
    #valid_loader = DataLoader(val_dataset, batch_size=EVAL_BATCH_SIZE, num_workers=0, pin_memory=PIN_MEMORY, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=EVAL_BATCH_SIZE, num_workers=0, pin_memory=PIN_MEMORY, shuffle=False, drop_last=False)
    #for datasss in train_loader:
    #    plt.imshow(datasss[1][0].permute(1,2,0))
    #    plt.show()
    #print(len(train_loader))
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
    #trainer.predict_epoch(test_loader, epoch_index=0, verbose=False, logging_interval=1)
    
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

        performance_recorder.weight_path = os.path.join(PERFORMANCE_RECORD_DIR, f'epoch{epoch_index}.pt')
        performance_recorder.save_weight()
    
    #trainer.predict_epoch(test_loader, epoch_index=0, verbose=False, logging_interval=1)
    
    
    # final_output_dir= 'C:/Users/Choi Jun Ho/ai_competition_hairstyle/baseline/results/train/20210627233220/'
    # last model save
    performance_recorder.weight_path = os.path.join(PERFORMANCE_RECORD_DIR, 'last.pt')
    performance_recorder.save_weight()
    

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
    