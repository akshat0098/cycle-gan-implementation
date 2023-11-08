import torch
import torch.nn as nn 
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torchvision 
from PIL import Image
import os
import random 

class ReplayBuffer():
    
    def __init__(self,max_size=50) :
        assert (max_size >0) , "Empty buffer"
        self.max_size = max_size
        self.data = []

    def push_and_pop(self,data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element,0) ## conver into row wise
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0,1) > 0.5:
                    i = random.randint(0,self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element 
                else:
                    to_return.append(element)
            
        return torch.autograd.Variable(torch.cat(to_return))
    

class MyDataset(Dataset):
    
    def __init__(self,data_dir,transform=None):
        self.transform = transform
        a = [os.path.join(os.path.join(data_dir,'A'),data) for data in os.listdir(os.path.join(data_dir,'A'))]
        b = [os.path.join(os.path.join(data_dir,'B'),data) for data in os.listdir(os.path.join(data_dir,'B'))]

        self.images = list(zip(a,b))

    def __len__(self):
        return len(self.images)
    
    def __getitem(self,idx):
        imageA = Image.open(self.images[idx][0])
        imageB = Image.open(self.images[idx][1])

        if self.transform != None:
            imageA = self.transform(imageA)
            imageB = self.transform(imageB)
        
        return imageA,imageB
    

class ResidualBlock(nn.Module):
    
    def __init__(self,in_channels,out_channels):
        super(ResidualBlock,self).__init__()

        self.sequnetial = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels,out_channels,3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels,out_channels,3),
            nn.BatchNorm2d(out_channels)
        )
      

    def forward(self,x):
        o = self.sequnetial(x)
        o +=x
        return o
    
class Generator(nn.Module):
    def __init__(self,num_res_blocks=9):
        super(Generator,self).__init__()
        layers = []
        ##initial

        self.initial_conv = [
            nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=7,
                stride=1,
                padding=3
            )
        ]

        layers +=self.initial_conv

        for i in range(2):
            layers += [
                nn.Conv2d(in_channels=(64*(i+1)),
                          out_channels=(128*(i+1)),
                          kernel_size=3,
                          stride=2,
                          padding=1
                          )
            ]

        for i in range(num_res_blocks):
            layers += [
                ResidualBlock(256,256)
            ]
        
        conv_tranposes = [nn.ConvTranspose2d(256//i,128//i,3,stride=2,padding=1,output_padding=1) for i in range(1,3) ]
        layers += conv_tranposes

        last_layers = [nn.Conv2d(64,3,7,stride=1,padding=3)]
        layers += last_layers

        self.model = nn.sequential(*layers)

    def forward(self,x):
        x = self.model(x)
        return x 
    

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()


        self.sequential =nn.Sequential(
            nn.Conv2d(3,64,4,stride=2,padding=1),
            nn.LeakyReLU(0.2,inplace=True),

            nn.Conv2d(64,128,4,stride=2,padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2,inplace=True),

            nn.Conv2d(128,256,4,stride=2,padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2,inplace=True),

            nn.Conv2d(256,512,4,stride=1,padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2,inplace=True),

            nn.Conv2d(512,1,4,stride=1,padding=1),
            nn.AvgPool2d(30)

        )


    def forward(self,x):
        return self.sequential(x)[:,0,0,0]
    
def cycle_consistency_loss(fake_x,real_x,fake_y,real_y):
    criterion = nn.L1Loss()
    loss  = criterion(fake_x,real_x) + criterion(fake_y,real_y)
    return loss

def adversial_loss(fake,real):
    return torch.nn.MSELoss()(fake,real)

def discriminator_loss(dis_fake,dis_real):
    loss = ( (dis_real - 1) ** 2 ) + (dis_fake **2)
    return loss

def criterion_identity_loss(fake,real):
    return torch.nn.L1Loss()(fake,real)
