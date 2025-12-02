import torch.nn.functional as F
import torch.nn as nn
from focal import FocalLoss
def compute_loss(target, target_class, out_p, out_c, args):
    
    regression_loss = F.mse_loss(out_p, target)

    classification_loss = F.binary_cross_entropy_with_logits(out_c, target_class)

    loss = regression_loss + args.alpha * classification_loss

    return loss, regression_loss, classification_loss
