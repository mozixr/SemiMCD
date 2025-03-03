# Model
import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.parametrizations import spectral_norm

class Dis(torch.nn.Module):
    def __init__(
        self,
        n_class = 3
    ) -> None:
        super(Dis,self).__init__()
 
        self.model = nn.Sequential(
            spectral_norm(nn.Conv2d(n_class, 8, 5, 2, 2, bias=True)),
            nn.BatchNorm2d(8, track_running_stats=True),
            nn.LeakyReLU(0.2, inplace=True),
#            
            spectral_norm(nn.Conv2d(8, 16, 5, 2, 2, bias=True)),
            nn.BatchNorm2d(16, track_running_stats=True),
            nn.LeakyReLU(0.2, inplace=True),
            
            spectral_norm(nn.Conv2d(16, 32, 5, 2, 2, bias=True)),
            nn.BatchNorm2d(32, track_running_stats=True),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(32, 64, 5, 2, 2, bias=True)),
            nn.BatchNorm2d(64, track_running_stats=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.regressor = nn.Sequential(
            nn.Linear(4096, 1),
            nn.Sigmoid()
        )


    def forward(self, img):
        B = img.size(0)
        x = self.model(img) # B, 128, W/16, H/16
        x = x.view(B, -1)

        x = self.regressor(x)
        return x


class DecoderBlock(nn.Module):
    """
    
    """
    def __init__(
        self, in_channels, 
        mid_channels, 
        out_channels, 
    ):
        super().__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels
    
        self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2, align_corners=True),
                nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(mid_channels),
            )
        self.conv2 = nn.Sequential(
                nn.Conv2d(mid_channels, out_channels, 3, 1, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        x=self.up(x)
        x=self.conv2(x)

        return x

class ResUnet(nn.Module):
    """

    """
    def __init__(
        self, 
        filters = [32,64,128,256,512],
        backbone = 'resnet50',
        teacher = False,
        n_class = 3
    ):
        super().__init__()
        self.n_class = n_class
        self.filters = filters
        self.backbone = backbone
        self.teacher = teacher

        if self.backbone=='resnet50':
#            self.resnet = models.resnet50(weights='IMAGENET1K_V2')
            self.resnet = models.resnet50()

        if self.teacher:
            self.firstconv = nn.Conv2d(
                in_channels=1, 
                out_channels=self.filters[0], 
                kernel_size=7 if self.teacher else 3, 
                stride=2, 
                padding=3, 
                bias=False)
            self.encoder1 = self.resnet.layer1
            self.encoder2 = self.resnet.layer2
            self.encoder3 = self.resnet.layer3
        else:
            self.firstconv = nn.Conv2d(
                in_channels=1,
                out_channels=self.filters[0], 
                kernel_size=3, 
                stride=2, 
                padding=1, 
                bias=False
            )
            self.encoder1 = nn.Sequential(
                    nn.Conv2d(
                        in_channels=self.filters[0],
                        out_channels=self.filters[1], 
                        kernel_size=1, 
                        stride=1, 
                        padding=0
                    ),
                    nn.BatchNorm2d(self.filters[1]),
                    nn.ReLU(inplace=False),
                    ) if self.teacher==False else self.resnet.layer1

            self.encoder2 = nn.Sequential(
                    nn.Conv2d(
                        in_channels=filters[1],
                        out_channels=filters[2], 
                        kernel_size=2, 
                        stride=2, 
                        padding=0
                    ),
                    nn.BatchNorm2d(filters[2]), 
                    nn.ReLU(inplace=False),
                    ) if self.teacher==False else self.resnet.layer2
    
            self.encoder3 = nn.Sequential(
                    nn.Conv2d(
                        in_channels=self.filters[2],
                        out_channels=self.filters[3], 
                        kernel_size=2, 
                        stride=2, 
                        padding=0
                    ),
                    nn.BatchNorm2d(self.filters[3]), 
                    nn.ReLU(inplace=False),
                    ) if self.teacher==False else self.resnet.layer3

        self.firstbn = nn.BatchNorm2d(self.filters[0]) # resnet.bn1
        self.firstrelu = nn.ReLU(inplace=False) # resnet.relu

        self.firstmaxpool = self.resnet.maxpool

        # decoder
        self.center = DecoderBlock(
            in_channels=self.filters[3], 
            mid_channels=self.filters[3]*4, 
            out_channels=self.filters[3]
        )
        self.decoder1 = DecoderBlock(
            in_channels=self.filters[3]+self.filters[2], 
            mid_channels=self.filters[2]*4, 
            out_channels=self.filters[2]
        )
        self.decoder2 = DecoderBlock(
            in_channels=self.filters[2]+self.filters[1], 
            mid_channels=self.filters[1]*4, 
            out_channels=self.filters[1]
        )
        self.decoder3 = DecoderBlock(
            in_channels=self.filters[1]+self.filters[0], 
            mid_channels=self.filters[0]*4, 
            out_channels=self.filters[0]
        )

        self.final = nn.Sequential(
            nn.Conv2d(
                in_channels=int(self.filters[0]), 
                out_channels=self.n_class, 
                kernel_size=3, 
                stride=1, 
                padding=1
            ),
            nn.ReLU()
        )

        self.e1_out_channels = int(self.filters[1]/8) \
            if self.teacher else int(self.filters[1]*8)
        self.e3_out_channels = int(self.filters[3]/8) \
            if self.teacher else int(self.filters[3]*8)
        
        self.e1out = nn.Sequential(
            nn.Conv2d(
                in_channels=self.filters[1], 
                out_channels=self.e1_out_channels, 
                kernel_size=5, 
                stride=1, 
                padding=2, 
                bias=False
            ),
            nn.BatchNorm2d(self.e1_out_channels),
            nn.ReLU(inplace=False),
        )

        self.e3out = nn.Sequential(
            nn.Conv2d(
                in_channels=self.filters[3], 
                out_channels=self.e3_out_channels, 
                kernel_size=5, 
                stride=1, 
                padding=2, 
                bias=False
            ),
            nn.BatchNorm2d(self.e3_out_channels),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        x = self.firstconv(x) 
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x_ = self.firstmaxpool(x) 
        e1 = self.encoder1(x_) 
        e2 = self.encoder2(e1) 
        e3 = self.encoder3(e2) 

        center = self.center(e3) 
        d2 = self.decoder1(torch.cat([center,e2],dim=1))
        d3 = self.decoder2(torch.cat([d2,e1], dim=1))
        d4 = self.decoder3(torch.cat([d3,x], dim=1))
       
        if self.teacher:
            return self.e1out(e1), e1, self.e3out(e3), e3, self.final(d4)
        else:
            return e1, self.e1out(e1), e3, self.e3out(e3), self.final(d4)
