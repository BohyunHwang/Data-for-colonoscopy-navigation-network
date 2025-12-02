import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from utils import *
from opt import parse_args
from model import gen_model
import train
import valid
from plot_prediction import predict
from scheduler import CosineAnnealingWarmUpRestarts
from contextlib import redirect_stdout
from torch.optim.lr_scheduler import ReduceLROnPlateau



def main(args):
    
    # Check GPU to use CUDA
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")   # use CPU or GPU
    model = gen_model(args, device)

    # Define optimizer      
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) 
    scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=args.epochs, T_mult=1, eta_max=1e-4,  T_up=1, gamma=0.5)
    
    # Load Data
    train_loader, valid_loader = load_data(args)
    
    # record training process
    epoch_train_losses = [] 
    epoch_valid_losses = []

    # Evaluate Best loss
    best_loss = 1e5
    
    # start training
    
    print("\nStart Training")
    for epoch in range(args.epochs):
        # train, valid model
        train_loss, train_reg_loss, train_class_loss, train_f1 = train.run(model, device, train_loader, optimizer, args)
        valid_loss, valid_reg_loss, valid_class_loss, valid_f1 = valid.run(model, device, valid_loader, args)
        
      
        try:
            lr = optimizer.param_groups[0]['lr']
            scheduler.step()
        except:
            pass
        
        best_loss = min(valid_loss, best_loss)
        
        if valid_loss == best_loss:
            os.makedirs(args.model_name, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.model_name,'best_model.pth'))  # save best model
            best_train_loss, best_train_reg_loss, best_train_class_loss, best_train_f1 = train_loss, train_reg_loss, train_class_loss, train_f1
            best_valid_loss, best_valid_reg_loss, best_valid_class_loss, best_valid_f1 = valid_loss, valid_reg_loss, valid_class_loss, valid_f1
        
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(args.model_name,'model_epoch_{}.pth'.format(epoch+1)))  # save best model

        print('Epoch: {} loss: {:.4e}, reg: {:.4e}, cls: {:.4e}, F1: {:.3f} / val_loss: {:.4e}, val_reg: {:.4e}, val_cls: {:.4e}, val_F1: {:.3f} / lr : {:.4e}'
            .format(epoch+1, train_loss, train_reg_loss, train_class_loss, train_f1, valid_loss, valid_reg_loss, valid_class_loss, valid_f1, lr))

        # save results
        epoch_train_losses.append(train_loss)
        epoch_valid_losses.append(valid_loss)

        # save all train valid results
        T = np.array(epoch_train_losses)
        V = np.array(epoch_valid_losses)

        save_loss_path = os.path.join(args.model_name, 'loss')
        os.makedirs(save_loss_path, exist_ok=True)
        np.save(os.path.join(save_loss_path,'CRNN_epoch_training_loss.npy'), T)
        np.save(os.path.join(save_loss_path,'CRNN_epoch_valid_loss.npy'), V)

    with open('result.csv','a') as file:
        file.write('{},{},{},{},{:.4e},{:.4e},{:.4e},{:.3f},{:.4e},{:.4e},{:.4e},{:.3f},'\
            .format(args.model, args.epochs, args.alpha, args.seq_len, best_train_loss, best_train_reg_loss, best_train_class_loss, best_train_f1, best_valid_loss, best_valid_reg_loss, best_valid_class_loss, best_valid_f1))

    # Plot Loss
    fig = plt.figure(figsize=(8, 4))
    plt.plot(np.arange(1, args.epochs + 1), T)  # train loss (on epoch end)
    plt.plot(np.arange(1, args.epochs + 1), V)  # valid loss (on epoch end)
    plt.title("model loss")
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend(['train', 'valid'], loc="upper left")
    
    title = os.path.join(save_loss_path,"fig_Loss_RCnet.png")
    plt.savefig(title, dpi=600)

 
if __name__=='__main__':
    
    args = parse_args()
    print('\nTraining Conditions')
    for arg in vars(args):
        print (arg, ' : ', getattr(args, arg))

    main(args)
    predict(args)
