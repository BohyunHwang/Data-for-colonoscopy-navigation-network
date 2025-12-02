import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from models import cnn
from opt import parse_args
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


# ---- Custom Regression Target for Grad-CAM ---- #
class RegressionOutputTarget:
    def __init__(self, index):
        self.index = index  # 0 -> x, 1 -> y

    def __call__(self, model_output):
        if model_output.dim() == 2:
            return model_output[:, self.index]
        elif model_output.dim() == 1:
            return model_output[self.index]
        else:
            raise ValueError(f"Unexpected output shape: {model_output.shape}")


# ---- Wrapper ---- #
class CNNWrapperReg_ResNet(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.resnet = model.resnet
        self.drop = model.drop
        self.regressor = model.regressor

    def forward(self, x):
        # CNN forward expects (B,1,C,H,W) sometimes → ensure shape
        if x.dim() == 4:
            x = x.unsqueeze(1)  # (B,C,H,W) → (B,1,C,H,W) if needed
        x = self.resnet(x[:, 0, ...])
        x = x.view(x.size(0), -1)
        x = F.relu(x, inplace=True)
        x = self.drop(x)
        out = self.regressor(x)   # (B,2) → x,y regression
        return out


# ---- Paths ---- #
image_folder = r"C:\Users\smero\Desktop\Learning\dataset\images\Sequence05"
model_path = r"C:\Users\smero\Desktop\Learning\saved_result\best_model.pth"
save_folder = r"C:\Users\smero\Desktop\Learning\grad-cam\gradcam_ResNet34_phantom_regression(@seq5)"
os.makedirs(save_folder, exist_ok=True)


# ---- Load Model ---- #
args = parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = cnn.CNN(args).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

wrapped_model = CNNWrapperReg_ResNet(model)

print(model.resnet)
target_layers = [model.resnet[7][-1]]   # final conv layer


# ---- Preprocessing ---- #
mean = [0.567, 0.390, 0.388]
std = [0.159, 0.155, 0.168]
input_size = args.input_size if hasattr(args, 'input_size') else 224

transform = A.Compose([
    A.Resize(input_size, input_size),
    A.Normalize(mean=mean, std=std),
    ToTensorV2()
])


# ---- Image List ---- #
image_files = sorted([
    os.path.join(image_folder, f)
    for f in os.listdir(image_folder)
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
])


# ---- Grad-CAM Loop ---- #
for img_path in image_files:
    frame = cv2.imread(img_path)
    if frame is None:
        continue

    name = os.path.splitext(os.path.basename(img_path))[0]

    transformed = transform(image=frame)
    input_tensor = transformed["image"].unsqueeze(0).to(device)

    # frame RGB for CAM overlay
    rgb_img = frame.astype(np.float32) / 255.0
    rgb_img_resized = cv2.resize(rgb_img, (input_size, input_size))


    # ======================================================
    # 1) CAM for X Regression Output
    # ======================================================
    cam_x = GradCAM(model=wrapped_model, target_layers=target_layers)
    cam_output_x = cam_x(input_tensor=input_tensor, targets=[RegressionOutputTarget(0)])  # list
    cam_x_map = cam_output_x[0]
    cam_image_x = show_cam_on_image(rgb_img_resized, cam_x_map, use_rgb=True)

    ## To save the CAM for the x coordinate, uncomment the corresponding line in the code
    # cv2.imwrite(
    #     os.path.join(save_folder, f"{name}_gradcam_x.jpg"),
    #     cv2.cvtColor(cam_image_x, cv2.COLOR_RGB2BGR)
    # )


    # ======================================================
    # 2) CAM for Y Regression Output
    # ======================================================
    cam_y = GradCAM(model=wrapped_model, target_layers=target_layers)
    cam_output_y = cam_y(input_tensor=input_tensor, targets=[RegressionOutputTarget(1)])
    cam_y_map = cam_output_y[0]
    cam_image_y = show_cam_on_image(rgb_img_resized, cam_y_map, use_rgb=True)

    ## To save the CAM for the y coordinate, uncomment the corresponding line in the code
    # cv2.imwrite(
    #     os.path.join(save_folder, f"{name}_gradcam_y.jpg"),
    #     cv2.cvtColor(cam_image_y, cv2.COLOR_RGB2BGR)
    # )


    # ======================================================
    # 3) Combined CAM (x and y)
    # ======================================================

    # A) x & y Average
    cam_xy = (cam_x_map + cam_y_map) / 2

    # B) Max_based
    # cam_xy = np.maximum(cam_x_map, cam_y_map)

    cam_image_xy = show_cam_on_image(rgb_img_resized, cam_xy, use_rgb=True)

    cv2.imwrite(
        os.path.join(save_folder, f"{name}_gradcam_xy.jpg"),
        cv2.cvtColor(cam_image_xy, cv2.COLOR_RGB2BGR)
    )

print(f"[Completed] The u, v CAM results have been successfully saved to {save_folder}.")