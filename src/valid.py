import torch
import torch.nn.functional as F
from custom_loss import compute_loss
from sklearn.metrics import f1_score

def run(model, device, valid_loader, args):
    # set model as validing mode
    model.eval()
    valid_loss = 0
    valid_reg_loss = 0
    valid_class_loss = 0
    
    all_y = []
    all_y_pred = []
    
    with torch.no_grad():
        for image, target, target_class in valid_loader:
            image = image.to(device)
            target = target.to(device)
            target_class = target_class.to(device)

            out_p, out_c = model(image)

            loss, reg_loss, class_loss = compute_loss(target, target_class, out_p, out_c, args)
            
            y_pred_tags = torch.round(torch.sigmoid(out_c))
        
            all_y.extend(target_class.cpu().detach().numpy().tolist())
            all_y_pred.extend(y_pred_tags.cpu().detach().numpy().tolist())
        
            valid_loss += loss.item() * image.size(0)
            valid_reg_loss += reg_loss.item() * image.size(0)
            valid_class_loss += class_loss.item() * image.size(0)

            
    valid_loss /= len(valid_loader.dataset)
    valid_reg_loss/= len(valid_loader.dataset)
    valid_class_loss/= len(valid_loader.dataset)
    valid_f1 = f1_score(all_y, all_y_pred)
    
    return valid_loss, valid_reg_loss, valid_class_loss, valid_f1
