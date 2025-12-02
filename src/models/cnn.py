import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

# 2D CNN encoder using ResNet-18 pretrained
class CNN(nn.Module):
    def __init__(self,args):

        super(CNN, self).__init__()
        self.fc_hidden1 = args.hidden1
        self.fc_hidden2 = args.hidden2
        self.drop_p = args.dropout
        self.output_size = args.output_size

        resnet = models.resnet34(pretrained=True)
        modules = list(resnet.children())[:-1]     
        self.resnet = nn.Sequential(*modules)
        
        self.drop = nn.Dropout(p = self.drop_p)
        self.regressor = nn.Linear(resnet.fc.in_features, self.output_size)
        self.classifier = nn.Linear(resnet.fc.in_features, 1)
        
    def forward(self, x):
        x = self.resnet(x[:,0,...])
        x = x.view(x.size(0),-1)
        x = F.relu(x, inplace = True)
        
        x = self.drop(x)
        class_num = self.classifier(x)
        coordinate = self.regressor(x)
        
        return coordinate.unsqueeze(1), class_num
    
## ---------------------- end of CRNN module ---------------------- ##