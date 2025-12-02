# encoding:utf-8
import torch, numpy as np
import torch.nn as nn

from models.resnet import resnet18
from torchsummary import summary


class FCN(nn.Module):
    
    def __init__(self, backbone, in_channel=512, num_classes=1):
        super(FCN, self).__init__()
        self.backbone = backbone
        self.cls_num = num_classes

        self.relu    = nn.ReLU(inplace=True)
        self.Conv1x1 = nn.Conv2d(in_channel, self.cls_num, kernel_size=1)
        
        self.bn1 = nn.BatchNorm2d(self.cls_num)
        self.bn2 = nn.BatchNorm2d(self.cls_num)
        self.bn3 = nn.BatchNorm2d(self.cls_num)
        self.bn4 = nn.BatchNorm2d(self.cls_num)
        self.bn5 = nn.BatchNorm2d(self.cls_num)
        
        self.DC1 = nn.ConvTranspose2d(self.cls_num, self.cls_num, kernel_size=4, stride=2, dilation=1, padding=1)
        self.DC2 = nn.ConvTranspose2d(self.cls_num, self.cls_num, kernel_size=4, stride=2, dilation=1, padding=1)
        self.DC3 = nn.ConvTranspose2d(self.cls_num, self.cls_num, kernel_size=4, stride=2, dilation=1, padding=1)
        self.DC4 = nn.ConvTranspose2d(self.cls_num, self.cls_num, kernel_size=4, stride=2, dilation=1, padding=1)
        self.DC5 = nn.ConvTranspose2d(self.cls_num, self.cls_num, kernel_size=4, stride=2, dilation=1, padding=1)


    def forward(self, x):
        x3, x4, x5 = self.backbone(x)
        
        x = self.Conv1x1(x5)
        
        x = self.bn1(self.relu(self.DC1(x)))
        x = self.bn2(self.relu(self.DC2(x)))
        x = self.bn3(self.relu(self.DC3(x)))
        x = self.bn4(self.relu(self.DC4(x)))
        x = self.bn5(self.relu(self.DC5(x)))
        
        x = torch.sigmoid(x) * 255
        
        return x 
    
if __name__ == "__main__":
    from torchvision import transforms, utils
    device = 'cuda:0'
    backbone = resnet18(pretrained = True, dilated = False)
    model = FCN(backbone, in_channel=512, num_classes=1)
    
    summary(model.to(device), (3, 256, 256))

    tf = transforms.Compose([
        transforms.RandomResizedCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.256, 0.225])
    ])

    in_img = np.zeros((256, 256, 3), np.uint8)
    t_img = transforms.ToTensor()(in_img)
    t_img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.256, 0.225])(t_img)
    t_img.unsqueeze_(0)
    t_img = t_img.to(device)
    
    x = model.forward(t_img)
    print(x.shape)


