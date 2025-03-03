import torch

###### validation 
from sklearn.metrics import precision_score, recall_score
import argparse
from statistics import mean
from DataLoader import Data_pipe
from tqdm import tqdm
from LossFunction import Loss_f
import matplotlib.pyplot as plt

def validation(Model, args, image_size):
    batch_size=1
    device = 'cuda'
    image_size = (image_size, image_size)

    if args == 'MontgomerySet':
        # Model = torch.load('MontgomerySet_Model_s')
        n_class = 3
        test_dataset = Data_pipe('../Dataset/MontgomerySet/test/',
                                  '../Dataset/MontgomerySet/label-3cls/',
                                 image_size=image_size,test=True)
    elif args == 'JSRT':
        # Model = torch.load('JSRT_Model_s')
        n_class = 3
        test_dataset = Data_pipe('../Dataset/JSRT/test/',
                                  '../Dataset/JSRT/label-3cls/',
                                 image_size=image_size,test=True)
    elif args == 'ShenZhen':
        # Model = torch.load('ShenZhen_Model_s')
        n_class = 2
        test_dataset = Data_pipe('../Dataset/ShenZhen/test/',
                                  '../Dataset/ShenZhen/label-2cls/',
                                 image_size=image_size,test=True)
        
    test_loader = torch.utils.data.DataLoader(test_dataset, 
                                              batch_size=batch_size, 
                                              shuffle=False)
    
    loss_sum = []
    loss_s = 0
    count = 0
    num = 1
    precision = []
    recall = []
    
    for index, (img,mask) in enumerate(test_loader):
        count += 1
        img = img.to(device)
        mask = mask.to(device)
        
        Model.eval()
        
        torch.no_grad()
        with torch.no_grad():
            
            e1l, e1h, e3l, e3h, output = Model(img)
            loss = Loss_f().diceCE_Loss(output, mask, dice=True, test=1, num_classes=n_class)
                
            loss_sum.append(loss)
            loss_s += loss
            
            labels = Loss_f().one_hot(mask, num_classes=n_class)
            precision.append(precision_score(
                                labels[:,:,:,:].flatten().cpu(),
                                torch.softmax(output, 1)[:,:,:,:].flatten().int().cpu())
                            )
            recall.append(recall_score(
                            labels[:,:,:,:].flatten().cpu(),
                            torch.softmax(output, 1)[:,:,:,:].flatten().int().cpu())
                         )
    
    print('Dice', loss_s/count)
    print('precision',mean(precision))
    print('recall',mean(recall))
