import os
import cv2
import fnmatch
from imageio.v2 import imread
import random
from shutil import copy

# JSRT
print('Processing JSRT')
# Convert to single channel
# ImgPath: images path
ImgPath = '../Dataset/JSRT/meta_image/'
NewImgPath = '../Dataset/JSRT/images/'

def Folder_detection(Path):
    if not os.path.exists(Path):
        os.makedirs(Path)

def ConvertSC(ImgPath, NewImgPath, new_lowest = 512):
    Folder_detection(NewImgPath)
    img_names = fnmatch.filter(sorted(os.listdir(ImgPath)),'*.png')
    for name in img_names:
        img = imread(ImgPath+name)
        
        width =img.shape[0] 
        height =img.shape[1] 
        #以下代码实现按设置好的最长边像素数缩放图片，并保持宽高比不变
        if(width>=height):
            longest=height
            new_width = int(width * new_lowest / height)
            out=cv2.resize(img,(new_width,new_lowest),cv2.INTER_AREA)
        else:
            longest=width
            new_height = int(height * new_lowest / width)
            out=cv2.resize(img,(new_lowest,new_height),cv2.INTER_AREA)
        try:
            if out.shape[2]!=1:
                cv2.imwrite(NewImgPath+name.split('.')[0]+'.png', out[:,:,0])
        except:
            cv2.imwrite(NewImgPath+name.split('.')[0]+'.png', out)
    return img_names
    
img_names = ConvertSC(ImgPath, NewImgPath)

# Extract labels
# LabelPath: label path
LabelPath = '../Dataset/JSRT/masks/'
NewLabelPath = '../Dataset/JSRT/label-3cls/'
Folder_detection(NewLabelPath)

label_names = fnmatch.filter(sorted(os.listdir(LabelPath)),'*.tif')

# generate 3-class labels
for name in label_names:
    img = imread(LabelPath+name)
 
    left_L = img[0]
    right_L = img[1]
    
    img = left_L*1+right_L*2
    cv2.imwrite(NewLabelPath+name.split('.')[0]+'.png', img)

# random create train and test set - resize to 512
def random_sample(train_path, test_path, NewImgPath):
    img_names = fnmatch.filter(sorted(os.listdir(NewImgPath)),'*.png')
    random.seed(1234567)
    train_name = random.sample(img_names, 2)
    test_name = [x for x in img_names if x not in train_name]
    
    Folder_detection(train_path)
    Folder_detection(test_path)
    
    for file in img_names:
        if file in train_name:
            copy(NewImgPath+file, train_path+file)
        else:
            copy(NewImgPath+file, test_path+file)
            
train_path = '../Dataset/JSRT/train/'
test_path = '../Dataset/JSRT/test/'
random_sample(train_path, test_path, NewImgPath)
print('JSRT Done')

# MontgomerySet
print('Processing MontgomerySet')
# Convert to single channel
ImgPath = '../Dataset/MontgomerySet/meta_image/'
NewImgPath = '../Dataset/MontgomerySet/images/'
img_names = ConvertSC(ImgPath, NewImgPath)

# combine lung mask
LlabelPath = '../Dataset/MontgomerySet/L_masks/'
RlabelPath = '../Dataset/MontgomerySet/R_masks/'
NewLabelPath = '../Dataset/MontgomerySet/label-3cls/'
Folder_detection(NewLabelPath)

for name in img_names:
    left_L = imread(LlabelPath+name)
    right_L = imread(RlabelPath+name)
    left_L=left_L.astype(int)
    right_L=right_L.astype(int)
    cv2.imwrite(NewLabelPath+name, left_L+right_L*2)

# random create train and test set
train_path = '../Dataset/MontgomerySet/train/'
test_path = '../Dataset/MontgomerySet/test/'
random_sample(train_path, test_path, NewImgPath)
print('MontgomerySet Done')

# ShenZhen
print('Processing ShenZhen')
# Convert to single channel
ImgPath = '../Dataset/ShenZhen/meta_image/'
NewImgPath = '../Dataset/ShenZhen/images/'
img_names = ConvertSC(ImgPath, NewImgPath)

# convert lung mask
LabelPath = '../Dataset/ShenZhen/masks/'
NewLabelPath = '../Dataset/ShenZhen/label-2cls/'
Folder_detection(NewLabelPath)
img_names = fnmatch.filter(sorted(os.listdir(LabelPath)),'*.png')

for name in img_names:
    label = imread(LabelPath+name)

    label=label.astype(int)
    label[label==255]=1
    cv2.imwrite(NewLabelPath+name.split('_mask')[0]+name.split('_mask')[1], label)

# random create train and test set
train_path = '../Dataset/ShenZhen/train/'
test_path = '../Dataset/ShenZhen/test/'
random_sample(train_path, test_path, NewImgPath)
print('ShenZhen Done')