# loss function
import torch 
import torch.nn as nn

class Loss_f():
    def diceCoeff(self, preds, gts, smooth=1e-5):
        self.N = gts.size(0)
        self.pred_flat = preds.view(self.N, -1)
        self.gt_flat = gts.view(self.N, -1)
    
        intersection = (self.pred_flat * self.gt_flat).sum(1)
        unionset = self.pred_flat.sum(1) + self.gt_flat.sum(1)
        loss = (2 * intersection + smooth) / (unionset + smooth)
        loss = loss.sum() / self.N
    
        return loss
    
    def one_hot(self, mask, num_classes=7, device='cuda'): 
        self.batch_size = mask.shape[0]
        self.h, self.w = mask.shape[2], mask.shape[3]
        self.temp_label = mask.squeeze(1) # b,256,256
        self.temp_targets = []
        for layer in range(num_classes):
            self.one = torch.ones(self.batch_size, self.h, self.w)
            self.one[self.temp_label != layer] = 0
            self.one = self.one.view(self.batch_size,1,self.h,self.w)
            self.temp_targets.append(self.one)
        self.temp_targets = torch.cat(self.temp_targets, dim=1)
        return self.temp_targets.to(device)
        
    def diceCE_Loss(
        self,
        inputs, 
        targets, 
        num_classes=3, 
        smooth=1e-5, 
        dice=False, 
        test=0,
        device = 'cuda'
    ):
    
        self.temp_inputs = torch.softmax(inputs, 1)
        self.targets = self.one_hot(targets, num_classes, device)
    
        self.CEntropy = nn.functional.log_softmax(inputs, dim = 1)
        self.celoss = torch.mean(-1*self.targets*self.CEntropy)
               
        self.class_dice = []
        for i in range(test, num_classes):
            self.class_dice.append(
                self.diceCoeff(self.temp_inputs[:, i:i + 1, :], 
                          self.targets[:, i:i + 1, :], 
                          smooth=1e-5)
            )
        self.mean_dice = sum(self.class_dice) / len(self.class_dice)
        
        if dice: return self.mean_dice
        else: return (1 - self.mean_dice) + self.celoss
    
    def BarlowTwins(self, student, teacher):
        self.stu = student+1e-5
        self.teac = teacher+1e-5
        self.loss = self.stu * (self.teac) / \
            torch.abs(self.stu) / torch.abs(self.teac)
        return self.loss.add_(-1).pow_(2).sum()