import os
import numpy as np
import pandas as pd

import cv2
from PIL import Image
import scipy.misc as misc
from imgaug import augmenters as iaa

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
    def __init__(self, 
                 df, 
                 img_dir, 
                 transform=None, 
                 mode='train', 
                 site=1, 
                 channels=[1,2,3,4,5,6]):
        self.records = df.to_records(index=False)
        self.channels = channels
        self.site = site
        self.mode = mode
        self.img_dir = img_dir
        self.len = len(df)
        self.transform = transform

    def __getitem__(self, index):
        experiment = self.records[index].experiment 
        plate = self.records[index].plate 
        well = self.records[index].well
    
        img_channels = [np.array(Image.open(os.path.join(self.img_dir, self.mode, experiment, f'Plate{plate}', f'{well}_s{self.site}_w{channel}.png')),  dtype=np.float32) for channel in range(1,7)]
        
        img = np.stack([channel for channel in img_channels],axis=2)

        if self.transform is not None:
            img = self.transform(image=img)['image']

        img = torch.from_numpy(img.transpose((2, 0, 1))).float()

        type_3_mean = [8.610294408432136, 20.315573365640237, 12.695983064895957, 12.45387823673216, 8.431911098390875, 15.246410689108854]
        type_3_std = [10.643278795499091, 13.019137335474026, 5.666254650446652, 7.522788892443989, 6.084863651635668, 7.774823745945344]

        img = T.Normalize(mean=type_3_mean, std=type_3_std)(img)        

        # img = T.Normalize(mean=[6.74696984, 14.74640167, 10.51260864, 10.45369445,  5.49959796, 9.81545561],
        #                   std=[7.95876312, 12.17305868, 5.86172946, 7.83451711, 4.701167, 5.43130431])(img)
            
        if self.mode == 'train':
            img = T.RandomErasing()(img)
            return img, self.records[index].id_code, int(self.records[index].sirna)
        else:
            return img, self.records[index].id_code            
        
    def __len__(self):
        return self.len


def test():
    size = 224
    resize = Resize(height=size, width=size, always_apply=True)

    tsfm = Compose([
        RandomRotate90(),
        resize
    ]) 

    df = pd.read_csv(os.path.join('data', 'U2OS_train_small.csv'))
    ds = DefaultDataset(df, 'data', transform=None, mode='train', site=1)

    print(ds[0][0].size())
    

if __name__ == '__main__':
    test()