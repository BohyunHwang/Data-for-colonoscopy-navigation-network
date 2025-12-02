import torch
import torch.nn.functional as F
from custom_loss import compute_loss
from sklearn.metrics import f1_score

def run(model, device, train_loader, optimizer, args):
    # set model as training mode
    model.train()
    train_loss = 0
    train_reg_loss = 0
    train_class_loss = 0
    train_f1 = 0
    
    all_y = []
    all_y_pred = []
    
    for image, target, target_class in train_loader:
        image = image.to(device)
        target = target.to(device)
        target_class = target_class.to(device)

        optimizer.zero_grad()
        
        out_p, out_c = model(image)
       
        y_pred_tags = torch.round(torch.sigmoid(out_c))

        all_y.extend(target_class.cpu().detach().numpy().tolist())
        all_y_pred.extend(y_pred_tags.cpu().detach().numpy().tolist())
        
        loss, reg_loss, class_loss = compute_loss(target, target_class, out_p, out_c, args)
        
        train_loss += loss.item() * image.size(0)
        train_reg_loss += reg_loss.item() * image.size(0)
        train_class_loss += class_loss.item() * image.size(0)

        loss.backward()
        optimizer.step()
        
    train_loss /= len(train_loader.dataset)
    train_reg_loss/= len(train_loader.dataset)
    train_class_loss/= len(train_loader.dataset)

    train_f1 = f1_score(all_y, all_y_pred)
    
    print('train_loss : ',train_loss)
    return train_loss, train_reg_loss, train_class_loss, train_f1
