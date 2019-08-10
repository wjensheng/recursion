import os
import numpy as np
import pandas as pd

import cv2
from PIL import Image
import scipy.misc as misc

import torch
from torchvision import transforms as T
from albumentations import Compose, RandomRotate90, Flip, Transpose, Resize, Normalize
from albumentations import RandomContrast, RandomBrightness, RandomGamma
from albumentations import Blur, MotionBlur, InvertImg
from albumentations import Rotate, ShiftScaleRotate, RandomScale
from albumentations import GridDistortion, ElasticTransform

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

class DefaultDataset(Dataset):
    
    def __init__(self, df, img_dir, transform=None, mode='train', site=1, channels=[1,2,3,4,5,6]):
        self.records = df.to_records(index=False)
        self.img_dir = img_dir
        self.transform = transform
        self.mode = mode
        self.site = site
        self.len = len(df)
        self.channels = channels

    def __getitem__(self, index):
        exp, plate, well = self.records[index].experiment, self.records[index].plate, self.records[index].well
    
        img_channels = []        
        for channel in range(1, 7):
            img_path = os.path.join(self.img_dir, self.mode, exp, f'Plate{plate}', f'{well}_s{self.site}_w{channel}.png')
            img = misc.imread(img_path)            
            img_channels += np.array(img, dtype=np.float32),
                
        one_img = np.stack([channel for channel in img_channels],axis=2)

        if self.transform is not None:
            img = self.transform(image=one_img)

        img = torch.from_numpy(img).float()

        if self.mode == 'train':
            return img, self.records[index].id_code, int(self.records[index].sirna)
        else:
            return img, self.records[index].id_code

    def __len__(self):
        return self.len


if __name__ == "__main__":
    df = pd.read_csv(os.path.join('data', 'U2OS_train_small.csv'))

    # base_aug = T.Compose([
    #     T.ToTensor(),
    #     # T.RandomRotation(degrees=(-90, 90)),
    #     T.RandomVerticalFlip(),
    #     T.RandomHorizontalFlip(),
    #     T.Resize((SIZE, SIZE)),        
    #     # T.Normalize(mean=MEAN, std=STD)
    # ])

    # base_aug = Compose([
    #     RandomRotate90(),
    #     Flip(),
    #     Transpose(),
    #     Resize(SIZE, SIZE, always_apply=True),        
    #     # Normalize(mean=MEAN, std=STD)
    # ])

    # base_aug = tta_transform()
    # ds = DefaultDataset(df, 'data', transform=base_aug, mode='train', site=1)
    # print(ds[0][0].size())