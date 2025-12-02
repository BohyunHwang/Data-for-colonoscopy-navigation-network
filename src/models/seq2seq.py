import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchvision.models import resnet18
import random

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

class EncoderRNN(nn.Module):
    def __init__(self, args):
        super(EncoderRNN, self).__init__()

        self.LSTM = nn.LSTM(
            input_size=512,
            hidden_size=args.h_RNN,
            num_layers=args.num_layers,
            batch_first = True
        )

    def forward(self, x):
        out, hidden = self.LSTM(x)

        return out, hidden
    
class DecoderRNN(nn.Module):
    def __init__(self, args):
        super(DecoderRNN, self).__init__()

        self.embedding = nn.Linear(args.output_size, 512),
                
        self.LSTM = nn.LSTM(
            input_size=512,
            hidden_size=args.h_RNN,
            num_layers=args.num_layers,
            batch_first = True
        )

        self.fc = nn.Linear(args.h_RNN, args.output_size)

    def forward(self, x, hidden):
        
        x = self.embedding(x).unsqueeze(1)
        x, hidden = self.LSTM(x, hidden)
        x = self.fc(x)
        
        return x, hidden
    
    
# Seq2Seq
class Seq2Seq(nn.Module):
    def __init__(self, args, device):
        super(Seq2Seq, self).__init__()

        self.cnn_encoder = EncoderCNN(args)
        self.rnn_encoder = EncoderRNN(args)
        self.rnn_decoder = DecoderRNN(args)
        self.device = device
        
    def forward(self, x, targets, teacher_forcing_ratio=1):
        
        x = self.cnn_encoder(x)
        
        # x.size() = [batch_size, seq_len, 512]
        
        batch_size, _, _ = x.size()
        
        _, target_len, input_size = targets.size()
        
        _, hidden = self.rnn_encoder(x)
        
        decoder_input = torch.zeros(batch_size, input_size).to(self.device)

        # decoder_input.size() = [batch_size, output_dim]
        
        outputs = torch.zeros(batch_size, target_len, input_size).to(self.device)

        for t in range(target_len): 

            out, hidden = self.rnn_decoder(decoder_input, hidden)
            # out.size() = [batch_size, 1, output_dim]
            
            out = out.squeeze(1)
            
            # out.size() = [batch_size, output_dim]
            if random.random() < teacher_forcing_ratio:
                decoder_input = targets[:, t, :]
            else:
                decoder_input = out
                
            outputs[:,t,:] = out

        return outputs