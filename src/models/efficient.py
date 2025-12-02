from torchsummary import summary
from efficientnet_pytorch import EfficientNet

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

# 2D CNN encoder using ResNet-18 pretrained
class EffiNet(nn.Module):
    def __init__(self,args):

        super(EffiNet, self).__init__()
        self.fc_hidden1 = args.hidden1
        self.fc_hidden2 = args.hidden2
        self.drop_p = args.dropout
        self.output_size = args.output_size

        self.effinet = EfficientNet.from_pretrained('efficientnet-b0')
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        
        self.regressor = nn.Sequential(
            nn.Linear(self.effinet._fc.in_features, self.fc_hidden1, bias = True),
            nn.BatchNorm1d(self.fc_hidden1),
            nn.ReLU(inplace=True),
            nn.Dropout(p = self.drop_p),
            nn.Linear(self.fc_hidden1, self.fc_hidden2),
            nn.BatchNorm1d(self.fc_hidden2),
            nn.ReLU(inplace=True),
            nn.Dropout(p = self.drop_p),
            nn.Linear(self.fc_hidden2, self.output_size)
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.effinet._fc.in_features, self.fc_hidden1, bias = True),
            nn.BatchNorm1d(self.fc_hidden1),
            nn.ReLU(inplace=True),
            nn.Dropout(p = self.drop_p),
            nn.Linear(self.fc_hidden1, self.fc_hidden2),
            nn.BatchNorm1d(self.fc_hidden2),
            nn.ReLU(inplace=True),
            nn.Dropout(p = self.drop_p),
            nn.Linear(self.fc_hidden2, 2)
            
        )
        
    def forward(self, x):
        x = self.effinet.extract_features(x[:,0,...])
        x = self.pool(x)

        features = x.view(x.size(0),-1)
        
        class_num = self.classifier(features)
        coordinate = torch.sigmoid(self.regressor(features))
        
        return coordinate.unsqueeze(1), class_num
    
## ---------------------- end of CRNN module ---------------------- ##
