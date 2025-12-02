
import os
import numpy as np
import torch
from torch.utils import data
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import transforms
from RCDataset import RC_Dataset
import pickle
## ---------------------- end of Dataloaders ---------------------- ##

def read_data(X_list, args):
    
    label_path = args.dataset + '/labels'
    
    df = pd.DataFrame()
    
    for exp in X_list:
        
        data = pd.read_csv(os.path.join(label_path, exp) + '.csv')
        data = data.replace('None', np.nan)
        data = data.dropna(axis=0)

        
        num_seq = len(data) - (args.seq_len-1)*args.step - args.target_len + 1
        #make sequences of images
        image_seq = [data[i:i+ (args.seq_len-1)*args.step+1 :args.step]['Path'].values for i in range(num_seq)]
        image_df = pd.DataFrame(data = image_seq)
        image_df.columns = ['image{}'.format(i) for i in range(args.seq_len)]
        
        #make sequences of data
        target_x_seq = [data[i+ (args.seq_len-1)*args.step: i+ (args.seq_len-1)*args.step + args.target_len]['X'].values for i in range(num_seq)]
        target_x_df = pd.DataFrame(data = target_x_seq)
        target_x_df.columns = ['X{}'.format(i) for i in range(args.target_len)]
        
        #make sequences of data
        target_y_seq = [data[i+ (args.seq_len-1)*args.step: i+ (args.seq_len-1)*args.step + args.target_len]['Y'].values for i in range(num_seq)]
        target_y_df = pd.DataFrame(data = target_y_seq)
        target_y_df.columns = ['Y{}'.format(i) for i in range(args.target_len)]
        
        #make sequences of class
        
        target_class_seq = [data[i+ (args.seq_len-1)*args.step: i+ (args.seq_len-1)*args.step + args.target_len]['Class'].values for i in range(num_seq)]
        target_class_df = pd.DataFrame(data = target_class_seq)
        target_class_df.columns = ['Class{}'.format(i) for i in range(args.target_len)]

        
        dataset = pd.concat([image_df, target_x_df, target_y_df, target_class_df], axis = 1)
        
        df = pd.concat([df, dataset])
        
    df.index = np.arange(0, len(df))
    
    #print(df)

    return df


def load_data(args):
    
    # path to images
    image_folder = os.path.join(args.dataset, 'images')

    # read all image folders and sort
    all_X_list = sorted(os.listdir(image_folder))
    print(all_X_list)
    # train, test split
    train_image_list, test_image_list = \
        train_test_split(all_X_list, test_size=0.2, random_state = 0)

    print(train_image_list, test_image_list)
    # make 
    train_dataset = read_data(train_image_list, args)
    test_dataset = read_data(test_image_list, args)
    
    print('Train_set : ', train_image_list)
    print('Test_set : ', test_image_list)
    print(len(train_dataset), len(test_dataset))

    # mean and standard deviation to normalize images
    mean, std = mean_std(args, train_dataset)
    
    # mean = [0.558,0.382,0.372]
    # std = [0.170,0.165,0.179]

    # size of inputs(images)
    res_size = args.input_size
    
    add_targets = {'image{}'.format(i) : 'image' for i in range(args.seq_len)}
    train_transform = A.Compose([
                                A.Resize(res_size,res_size),
                                A.VerticalFlip(p=0.5),
                                A.HorizontalFlip(p=0.5),
                                A.Rotate(45, p = 0.5, border_mode= 1),
                                A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5, p=0.5),
                                # A.ColorJitter(brightness=1, contrast=1, saturation=1, hue=0.5, p=1),
                                # A.RandomBrightnessContrast(brightness_limit=2, contrast_limit=2, p=1),
                                A.ToGray(p=0.5),  # Added grayscale transformation
                                A.Normalize(mean = mean, std= std),
                                transforms.ToTensorV2(),
                                ],
                                additional_targets= add_targets,
                                keypoint_params= A.KeypointParams(format = 'xy', remove_invisible = False))
        
    test_transform = A.Compose([ A.Resize(res_size,res_size),
                            A.Normalize(mean = mean, std= std),
                            transforms.ToTensorV2()],
                            additional_targets= add_targets,
                            keypoint_params= A.KeypointParams(format = 'xy', remove_invisible = False))
    
    train_params = {'batch_size': args.batch_size, 'shuffle': True, 'num_workers': 4, 'pin_memory': True, 'drop_last': True} \
        if not args.no_cuda else {}

    test_params = {'batch_size': args.batch_size, 'shuffle': True, 'num_workers': 4, 'pin_memory': True, 'drop_last': False} \
        if not args.no_cuda else {}
        
    # Datasets
    train_set = RC_Dataset(train_dataset, train_transform, args)
    test_set = RC_Dataset(test_dataset, test_transform, args)

    # Dataloaders
    train_loader = data.DataLoader(train_set, **train_params)
    test_loader = data.DataLoader(test_set, **test_params)
    
    
    return train_loader, test_loader


# extract mean and std for images
def mean_std(args, dataset):
    
    rgb_mean_std_filename = os.path.join(args.dataset, 'rgb_mean_std.pkl')
    
    if os.path.isfile(rgb_mean_std_filename ):
        print("\nRGB data found!")
        with open(rgb_mean_std_filename, 'rb') as fr:
            data = pickle.load(fr)
            
        mean = data['mean']
        std = data['std']
        
        print('mean : {:.3f},{:.3f},{:.3f}'.format(mean[0],mean[1],mean[2]))
        print('std : {:.3f},{:.3f},{:.3f}\n'.format(std[0],std[1],std[2]))

        
    else:
        print("\nNo RGB data found!")
        mean = 0
        std = 0
        num = 0
        
        image_list = dataset['image0'].values
        num = len(image_list)
            
        for frame in dataset['image0']:
            frame = os.sep.join(frame.split("/")[-2:])
            image = cv2.imread(args.dataset + '/images/' + frame)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
            img_mean, img_std = cv2.meanStdDev(image)
            
            mean += img_mean / 255
            std += img_std / 255
                
        mean /= num
        std /= num
        
        mean = mean.reshape(-1).tolist()
        std = std.reshape(-1).tolist()
        
        data = {"mean" : mean, "std" : std}
        
        with open(rgb_mean_std_filename, 'wb') as fr:
            pickle.dump(data, fr)
        
        print('mean : {:.3f},{:.3f},{:.3f}'.format(mean[0],mean[1],mean[2]))
        print('std : {:.3f},{:.3f},{:.3f}\n'.format(std[0],std[1],std[2]))

    return mean, std
