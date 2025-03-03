# Dataloader
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose, Resize
from torchvision.transforms import RandomCrop, ColorJitter
import os
import numpy as np
import torch
import torchvision.transforms.functional as f

class Data_pipe(torch.utils.data.Dataset):
    def __init__(self, filepath, labelpath, image_size, test=False, semi=False):

        self.test = test
        self.semi = semi
        self.filepath = filepath
        self.labelpath = labelpath
        self.image_size = image_size
        self.data_info = \
            [self.filepath + str(i) for i in os.listdir(self.filepath)]
        self.label_info = \
            [self.labelpath + str(i) for i in os.listdir(self.filepath)]

        # label
        self.transforms_l = Compose([
            Resize(image_size, interpolation=f._interpolation_modes_from_int(0)),
            RandomCrop(size=self.image_size),
            ToTensor()])
        # teacher
        self.transforms_t = Compose([
            Resize(image_size, interpolation=f._interpolation_modes_from_int(0)), 
            RandomCrop(size=self.image_size),
            ColorJitter(0.1,0.1),
            ToTensor()])
        # student
        if self.test:
            self.transforms_s = Compose([
                Resize(self.image_size, 
                       interpolation=f._interpolation_modes_from_int(0)), 
                ToTensor()])
        else:
            self.transforms_s = Compose([
                Resize(self.image_size, 
                       interpolation=f._interpolation_modes_from_int(0)), 
                RandomCrop(size=self.image_size),
                ToTensor()])
            
    def __len__(self):
        return len(os.listdir(self.filepath))
    def __getitem__(self,index):
        self.image = Image.open(self.data_info[index])

        self.seed = np.random.randint(2147483647)  # Random seed

        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        
        s_image = self.transforms_s(self.image) # student
        t_image = self.transforms_t(self.image) # teacher

        # semi用于参加一致性约束的无标签数据
        if self.semi!=True:
            self.label = Image.open(self.label_info[index])
            label = self.transforms_l(self.label)
        
        if self.test and self.semi==False: return s_image, label*255
        elif self.test==False and self.semi==False: 
            return t_image, s_image, label*255
        elif self.test==False and self.semi==True: return t_image, s_image
