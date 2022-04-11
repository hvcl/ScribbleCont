import os
import numpy as np
import warnings
import albumentations as A
from PIL import Image
from collections import defaultdict
from torch.utils.data import Dataset
from albumentations.pytorch.transforms import ToTensorV2


warnings.filterwarnings("ignore")

def get_transforms(input_size=256, jittering=False, need=('train', 'val')):
    transformations = {}
    if 'train' in need:
        if jittering==True:
            transformations['train'] = A.Compose([
                A.RandomCrop(height=input_size, width=input_size, p=1.),
                A.ShiftScaleRotate(p=0.7),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ColorJitter(p=0.5),
                A.Normalize(p=1.0),
                ToTensorV2(p=1.0),
            ], p=1.0, additional_targets={'scr': 'mask', 'weight': 'mask'})
        else:
            transformations['train'] = A.Compose([
                A.RandomCrop(height=input_size, width=input_size, p=1.),
                A.ShiftScaleRotate(p=0.7),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.Normalize(p=1.0),
                ToTensorV2(p=1.0),                
            ], p=1.0, additional_targets={'scr': 'mask', 'weight': 'mask'})
    if 'val' in need:
        transformations['val'] = A.Compose([
            A.Normalize(p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.0, additional_targets={'scr': 'mask', 'weight': 'mask'})
    return transformations


class dsbTrainDataset(Dataset):
    def __init__(self, data_folder, scr_folder, mask_folder, df, tfms):
        self.images = defaultdict(dict)
        for idx, (image_id, _) in df.iterrows():
            img = np.array(Image.open(os.path.join(data_folder, f'{image_id}.png')).convert('RGB'))
            scr = np.array(Image.open(os.path.join(scr_folder, f'{image_id}.png')))
            mask = (np.array(Image.open(os.path.join(mask_folder, f'{image_id}.png')).convert('L')) > 0)
            h, w = mask.shape
            h = (h // 32) * 32
            w = (w // 32) * 32
            self.images[idx]['id'] = image_id
            self.images[idx]['image'] = img[:h, :w, :]
            self.images[idx]['mask'] = mask[:h, :w].astype('uint8')
            self.images[idx]['scr'] = scr[:h, :w].astype('uint8')
            self.images[idx]['weight'] = np.zeros((h, w, 1), dtype=np.float32)
        self.tfms = tfms
        self.length = len(df)

    def __getitem__(self, idx):
        image_id = self.images[idx]['id']
        image = self.images[idx]['image']
        scribble = self.images[idx]['scr']
        weight = self.images[idx]['weight']
        mask = self.images[idx]['mask']

        if self.tfms:
            augmented = self.tfms(image=image,
                                  mask=mask,
                                  scr=scribble,
                                  weight=weight)
            image, scribble, weight, mask = augmented['image'], augmented['scr'], augmented['weight'], augmented['mask']   

        return image, scribble, weight, mask, image_id

    def __len__(self):
        return self.length


class dsbValidDataset(Dataset):
    def __init__(self, data_folder, mask_folder, df, tfms):
        self.images = defaultdict(dict)
        for idx, (image_id, _) in df.iterrows():
            img = np.array(Image.open(os.path.join(data_folder, f'{image_id}.png')).convert('RGB'))
            mask = (np.array(Image.open(os.path.join(mask_folder, f'{image_id}.png')).convert('L')) > 0)
            h, w = mask.shape
            h = (h // 32) * 32
            w = (w // 32) * 32
            self.images[idx]['id'] = image_id
            self.images[idx]['image'] = img[:h, :w, :]
            self.images[idx]['mask'] = mask[:h, :w].astype('uint8')
            self.images[idx]['weight'] = np.zeros((h, w, 1), dtype=np.float32)
        self.tfms = tfms
        self.length = len(df)

    def __getitem__(self, idx):
        image_id = self.images[idx]['id']
        image = self.images[idx]['image']
        weight = self.images[idx]['weight']
        mask = self.images[idx]['mask']

        augmented = self.tfms(image=image,
                                mask=mask,
                                weight=weight)
        image, weight, mask = augmented['image'], augmented['weight'], augmented['mask']
        return image_id, image, mask

    def __len__(self):
        return self.length


class dsbTestDataset(Dataset):
    def __init__(self, data_folder, mask_folder, df, tfms, monu=False):
        self.data_folder = data_folder
        self.mask_folder = mask_folder
        self.ImageIDs = df.ImageID.values
        self.tfms = tfms
        self.monu = monu

    def __getitem__(self, idx):
        image_id = self.ImageIDs[idx]
        image = np.array(Image.open(os.path.join(self.data_folder, f'{image_id}.png')).convert('RGB'))
        if self.monu==True:
            mask = np.array(Image.open(os.path.join(self.mask_folder, f'{image_id}.png')), dtype='int16')
        else:
            mask = np.array(Image.open(os.path.join(self.mask_folder, f'{image_id}.png')).convert('L'))
        
        h, w = mask.shape
        h = (h // 32) * 32
        w = (w // 32) * 32

        image = image[:h, :w, :]
        mask = mask[:h, :w]            

        augmented = self.tfms(image=image, mask=mask)
        image, mask = augmented['image'], augmented['mask']

        return image_id, image, mask

    def __len__(self):
        return len(self.ImageIDs)