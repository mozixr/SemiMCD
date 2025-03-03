# main 
import argparse
import torch
import torch.nn as nn
from itertools import cycle
import sys
import math
from tqdm import tqdm
from torch.utils.data import DataLoader
from ModelBuild import ResUnet
from ModelBuild import Dis
from LossFunction import Loss_f
from DataLoader import Data_pipe
from val import validation
import torch.nn.functional as F

def main(args):
    device = 'cuda'
    min_loss = args.min_loss
    num_epochs = args.num_epochs
    lr_min = args.lr_min
    lr_max = args.lr_max
    image_size = (args.image_size, args.image_size)
    warm_up = args.warm_up
    
    if args.dataset == 'MontgomerySet':
        print('MontgomerySet training')
        n_class = 3
        filepath = '../Dataset/MontgomerySet/'
        labelpath = '../Dataset/MontgomerySet/label-3cls/'
        train_b_size = 2
        semi_b_size = 16
        min_loss = 2e-1
    elif args.dataset == 'JSRT':
        print('JSRT training')
        n_class = 3
        filepath = '../Dataset/JSRT/'
        labelpath = '../Dataset/JSRT/label-3cls/'
        train_b_size = 2
        semi_b_size = 16
        min_loss = 2e-1
    elif args.dataset == 'ShenZhen':
        print('ShenZhen training')
        n_class = 2
        filepath = '../Dataset/ShenZhen/'
        labelpath = '../Dataset/ShenZhen/label-2cls/'
        train_b_size = 2
        semi_b_size = 16
        min_loss = 2e-1
    multiple = semi_b_size // train_b_size
    
    Model = ResUnet(filters=[16,32,64,128], n_class=n_class).to(device)
    T_Model = ResUnet(filters=[64,256,512,1024], 
                      n_class=n_class, teacher=True).to(device)
    Dis_t = Dis(n_class=n_class).to(device)
    
    train_dataset = Data_pipe(filepath+'train/',
                              labelpath,
                              image_size=image_size)
    train_loader = DataLoader(train_dataset, batch_size=train_b_size, shuffle=True)
    semi_dataset = Data_pipe(filepath+'test/',
                             labelpath,
                             image_size=image_size,semi=True)
    semi_loader = DataLoader(semi_dataset, batch_size=semi_b_size, shuffle=True)

    opt_d = torch.optim.Adam(Dis_t.parameters(), weight_decay=1e-6, lr=lr_max/500)
    
    optimizer = torch.optim.AdamW( 
            Model.parameters(), 
            lr=lr_max, 
            betas=(0.9, 0.95), 
            weight_decay=0.05 
        ) 
    T_optimizer = torch.optim.AdamW( 
            T_Model.parameters(), 
            lr=lr_max, 
            betas=(0.9, 0.95), 
            weight_decay=0.05 
        ) 
    lr_func = lambda epoch: min((lr_max * epoch / warm_up),
        (lr_min + (lr_max-lr_min)*(1 + math.cos(math.pi * \
        (epoch - warm_up) / (num_epochs - warm_up))) / 2))
    
    T_lr_func = lambda epoch: min((lr_max * epoch / warm_up),
        (lr_min + (lr_max-lr_min)*(1 + math.cos(math.pi * \
        (epoch - warm_up) / (num_epochs - warm_up))) / 2))
    
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, 
                                                     lr_lambda=lr_func)
    T_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(T_optimizer, 
                                                       lr_lambda=T_lr_func)

    ce = nn.BCELoss().to(device)
    lf = Loss_f()
    
    for epoch in range(num_epochs):

        losses = []
        
        step_count = 0
        
        for index, data in enumerate(tqdm(zip(cycle(train_loader), 
                                              semi_loader), 
                                          total=len(semi_loader))):
            step_count += 1
            img = data[1][1].to(device)
            t_img = data[1][0].to(device)
     
            Model.train()
            T_Model.train()

            # no-label stage
            for p in Dis_t.parameters():
                p.requires_grad = False
#
            e1l, e1h, e3l, e3h, output = Model(img)
            e1lr, e1hr, e3lr, e3hr, toutput = T_Model(t_img)
            fake_validity = Dis_t(toutput)
            
            constrain_loss = torch.mean(
                    -1*F.softmax(output,dim=1) * \
                    F.log_softmax(toutput,dim=1)
                ) + \
                torch.mean(
                    -1*F.softmax(toutput,dim=1) * \
                    F.log_softmax(output,dim=1)
                ) + \
                lf.BarlowTwins(e1l,e1lr) + \
                lf.BarlowTwins(e1h,e1hr) + \
                lf.BarlowTwins(e3l,e3lr) + \
                lf.BarlowTwins(e3h,e3hr) + \
                0.05*ce(fake_validity, torch.ones_like(fake_validity))

            optimizer.zero_grad()
            T_optimizer.zero_grad()
            constrain_loss.backward()
            optimizer.step()
            T_optimizer.step()       
    
            # Label stage
            for p in Dis_t.parameters():
                p.requires_grad = True

            img2 = data[0][1].to(device)
            t_img2 = data[0][0].to(device)
            mask = data[0][2].to(device)
            
            e1l, e1h, e3l, e3h, output = Model(img2)
            e1lr, e1hr, e3lr, e3hr, toutput = T_Model(t_img2)
            real_validity = Dis_t(lf.one_hot(mask, n_class))
            fake_validity = Dis_t(toutput.detach())
    
            semi_loss1 = torch.mean(
                    -1*F.softmax(output,dim=1)*\
                    F.log_softmax(toutput,dim=1)
                )
            semi_loss2 = torch.mean(
                    -1*F.softmax(toutput,dim=1)*\
                    F.log_softmax(output,dim=1)
                )

            loss = lf.diceCE_Loss(output, mask, num_classes=n_class) + \
                lf.diceCE_Loss(toutput, mask, num_classes=n_class)
            
            label_loss = multiple*(
                    loss + semi_loss1 + semi_loss2 + \
                    lf.BarlowTwins(e1l,e1lr) + \
                    lf.BarlowTwins(e1h,e1hr) + \
                    lf.BarlowTwins(e3l,e3lr) + \
                    lf.BarlowTwins(e3h,e3hr) + \
                    0.05*ce(fake_validity, torch.zeros_like(fake_validity)) + \
                    0.05*ce(real_validity, torch.ones_like(real_validity))
                )
            optimizer.zero_grad()
            T_optimizer.zero_grad()
            opt_d.zero_grad()
            label_loss.backward()
            optimizer.step()
            T_optimizer.step()
            opt_d.step()

            losses.append(label_loss.item()+constrain_loss.item())
      
        lr_scheduler.step()
        T_lr_scheduler.step()
        avg_loss = sum(losses) / len(losses)
        print(f'In epoch {epoch+1}, average traning loss is {avg_loss}.\t'+
             'lr= ' +str(optimizer.state_dict()['param_groups'][0]['lr'])
             ) 
        if avg_loss<min_loss and epoch > num_epochs - 100:
            optimizer.zero_grad()
            validation(Model, args.dataset, args.image_size)
            optimizer.zero_grad()
#            torch.save(Model, args.dataset + '_Model_s')
#            torch.save(T_Model, args.dataset+ '_Model_t')
            min_loss=avg_loss
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default = None)
    parser.add_argument('--num_epochs', type=int, default = 500)
    parser.add_argument('--lr_min', type=float, default = 1e-8)
    parser.add_argument('--lr_max', type=float, default = 0.05)
    parser.add_argument('--image_size', type=int, default = 128)
    parser.add_argument('--warm_up', type=int, default = 20)
    parser.add_argument('--min_loss', type=float, default = 5e-3)
    args = parser.parse_args()
    main(args)
