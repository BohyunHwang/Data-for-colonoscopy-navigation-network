from models import cnn, cnnlstm

def gen_model(args, device):
   
    assert args.model in [
        'cnn',
        'seq2seq',
    ]
    
    if args.model == 'cnn':
        model = cnn.CNN(args)

    elif args.model == 'seq2seq':
        print('\n## Importing Seq2Seq model')
        model = cnnlstm.CNNLSTM(args)        
        
    return model.to(device)
        