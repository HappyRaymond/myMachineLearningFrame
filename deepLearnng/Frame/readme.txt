# library
# standard library
import os

# third-party library
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt



class CNN1D(nn.Module):
    
    def setLayer(self,in_channel_num = 1,out_channel_num = 16,kernel_size = 3,stride=1 ,padding=1, Pool_kernel_size = 1 ,dropout = 0):
        Layer = []
        Layer.append(nn.Conv1d(
                in_channels=in_channel_num,              # input height
                out_channels=out_channel_num,            # n_filters
                kernel_size=kernel_size,              # filter size
                stride=stride,                   # filter movement/step
                padding=padding,                  # 
        ))
        
        Layer.append(nn.BatchNorm1d(out_channel_num))
        Layer.append(nn.ReLU(True))
        if(dropout > 0 ):
            Layer.append(nn.Dropout(dropout))
        if(Pool_kernel_size > 1):
            Layer.append(nn.MaxPool1d(kernel_size=Pool_kernel_size))
        return  Layer 
    
    
    def __init__(self):
        super(CNN1D, self).__init__()
        self.Conv1 = nn.Sequential(*self.setLayer(in_channel_num = 1,out_channel_num = 16,kernel_size = 3,stride=1 ,padding=1,Pool_kernel_size = 1 ,dropout = 0.4) )
        self.Conv2 = nn.Sequential(*self.setLayer(in_channel_num = 16,out_channel_num = 32,kernel_size = 3,stride=1 ,padding=1,Pool_kernel_size = 1 ,dropout = 0.4) )
        self.Conv3 = nn.Sequential(*self.setLayer(in_channel_num = 32,out_channel_num = 64,kernel_size = 3,stride=1 ,padding=1,Pool_kernel_size = 1 ,dropout = 0.4) )
        self.out = nn.Sequential(
            nn.Linear(13 * 64, 1)
        )
        self.residual_1 = nn.Sequential(*self.setLayer(in_channel_num = 1,out_channel_num = 16,kernel_size = 1,stride=1 ,padding=0,Pool_kernel_size = 1 ,dropout = 0) )
        self.residual_2 = nn.Sequential(*self.setLayer(in_channel_num = 16,out_channel_num = 32,kernel_size = 1,stride=1 ,padding=0,Pool_kernel_size = 1 ,dropout = 0) )
        
    def forward(self,x):
        x = x.reshape(x.shape[0],1,x.shape[1])
        
        
        residual = x
        x = self.Conv1(x)
        # residual= self.residual_1(residual)
        # x =x + residual
        x = self.Conv2(x)
        # residual= self.residual_2(residual)
        # x =x + residual
        x = self.Conv3(x)
        x = x.view(x.shape[0], -1)

        out = self.out(x)
        return out