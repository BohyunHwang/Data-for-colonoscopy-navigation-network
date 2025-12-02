import torch
from main import main
from opt import parse_args
from plot_prediction import predict
import numpy as np
for i in [1]:
    for j in np.arange(0.1,0.2,0.1):
        for t in range(1,11):

                args = parse_args()
                
                args.alpha = j
                args.save_model_path = 'seq_len_{}_alpha_{}_train_{}'.format(i, args.alpha, t)
                
                print('\nTraining Conditions')
                
                for arg in vars(args):
                    print (arg, ' : ', getattr(args, arg))
                
                main(args)
                predict(args)