import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchvision.models import resnet18

class EncoderCNN(nn.Module):
    def __init__(self,args):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet34(pretrained=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        
    def forward(self, x_3d):
        cnn_embed_seq = []
        
        for t in range(x_3d.size(1)):
            x = self.resnet(x_3d[:, t, :, :, :]) 
            x = x.view(x.size(0), -1) 
            cnn_embed_seq.append(x)

        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=1)
        
        return cnn_embed_seq


class DecoderRNN(nn.Module):
    def __init__(self, args):
        super(DecoderRNN, self).__init__()

        self.fc_hidden1 = args.hidden1
        self.fc_hidden2 = args.hidden2
        self.drop_p = args.dropout
        self.output_size = args.output_size
        
        self.LSTM = nn.LSTM(
            input_size=args.CNN_embed_dim,
            hidden_size=args.h_RNN,
            num_layers=args.num_layers,
            batch_first = True
        )

        self.regressor = nn.Sequential(
            nn.Linear(args.h_RNN, self.fc_hidden1, bias = True),
            nn.ReLU(inplace=True),
            nn.Dropout(p = self.drop_p),
            nn.Linear(self.fc_hidden1, self.fc_hidden2),
            nn.ReLU(inplace=True),
            nn.Dropout(p = self.drop_p),
            nn.Linear(self.fc_hidden2, self.output_size)
        )
        self.classifier = nn.Sequential(
            nn.Linear(args.h_RNN, self.fc_hidden1, bias = True),
            nn.ReLU(inplace=True),
            nn.Dropout(p = self.drop_p),
            nn.Linear(self.fc_hidden1, self.fc_hidden2),
            nn.ReLU(inplace=True),
            nn.Dropout(p = self.drop_p),
            nn.Linear(self.fc_hidden2, 2)
        )
            

    def forward(self, x):
        self.LSTM.flatten_parameters()
        out, _ = self.LSTM(x)
        
        class_num = self.classifier(out[:,-1,:])
        coordinate = torch.sigmoid(self.regressor(out[:,-1,:]))
        
        return coordinate, class_num

class CNNLSTM(nn.Module):
    def __init__(self, args):
        super(CNNLSTM, self).__init__()
        self.cnn_encoder = EncoderCNN(args)
        self.rnn_decoder= DecoderRNN(args)
        
    def forward(self, x):
        x = self.cnn_encoder(x)
        
        coordinate, class_num = self.rnn_decoder(x)
        
        return coordinate.unsqueeze(1), class_num