from unicodedata import bidirectional
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchvision.models import resnet18
from torch.nn.init import kaiming_normal_, orthogonal_

def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1, dropout=0):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout)#, inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout)#, inplace=True)
        )

class EncoderCNN(nn.Module):
    def __init__(self,args):
        super(EncoderCNN, self).__init__()
        
        self.batchNorm = True
        self.conv1   = conv(self.batchNorm,   6,   64, kernel_size=7, stride=2)
        self.conv2   = conv(self.batchNorm,  64,  128, kernel_size=5, stride=2)
        self.conv3   = conv(self.batchNorm, 128,  256, kernel_size=5, stride=2)
        self.conv3_1 = conv(self.batchNorm, 256,  256, kernel_size=3, stride=1)
        self.conv4   = conv(self.batchNorm, 256,  512, kernel_size=3, stride=2)
        self.conv4_1 = conv(self.batchNorm, 512,  512, kernel_size=3, stride=1)
        self.conv5   = conv(self.batchNorm, 512,  512, kernel_size=3, stride=2)
        self.conv5_1 = conv(self.batchNorm, 512,  512, kernel_size=3, stride=1)
        self.conv6   = conv(self.batchNorm, 512, 1024, kernel_size=3, stride=2)
        
        self.pooling = nn.AdaptiveAvgPool2d((1,1))
        self.rnn = nn.LSTM(
            input_size = 1024,
            hidden_size = 256,
            num_layers = 2,
            batch_first = True)
        self.rnn_drop_out = nn.Dropout(args.dropout)
        
        self.regressor = nn.Linear(256, 2)
        self.classifier = nn.Linear(256, 1)

        
    def forward(self, x):
        # print("1",x.shape)
        x = torch.cat(( x[:, :-1], x[:, 1:]), dim=2)
        # print("2",x.shape)
        batch_size = x.size(0)
        # print("batch_size",batch_size)
        seq_len = x.size(1)
        # print("seq_len ",seq_len )
        # CNN
        x = x.view(batch_size*seq_len, x.size(2), x.size(3), x.size(4))
        x = self.encode_image(x)
        x = self.pooling(x)
        x = x.view(batch_size, seq_len, -1)
        
        out, hc = self.rnn(x)
        out = self.rnn_drop_out(out[:,-1,:])
        pred = self.regressor(out)
        collision = self.classifier(out)
        
        return pred.unsqueeze(1), collision
    

    def encode_image(self, x):
        out_conv2 = self.conv2(self.conv1(x))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6(out_conv5)
        return out_conv6