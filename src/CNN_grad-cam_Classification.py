import os
import cv2
import torch
import numpy as np
import torch.nn as nn
from models import cnn
from opt import parse_args
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam import GradCAMPlusPlus

# ---- Custom target function for classification Grad-CAM ---- #
class ClassificationOutputTarget:
    def __call__(self, model_output):
        if model_output.dim() == 1:
            return model_output[0]
        else:
            return model_output[:, 0]

# ---- Wrapper class for passing classification output to Grad-CAM ---- #
class CNNWrapperCls(nn.Module):  # For collision probability classification
    def __init__(self, model):
        super().__init__()
        self.model = model

    # def forward(self, x):
    #     _, cls = self.model(x)
    #     return cls  # [batch, 1]

    def forward(self, x):
        # CNN model sometimes expects (B, seq_len, C, H, W) → Grad-CAM sends (B, C, H, W)
        if x.dim() == 4:
            x = x.unsqueeze(1)  # Convert to (B,1,C,H,W)
        _, cls = self.model(x)
        return cls


# ---- Paths ---- #
image_folder = r"C:\Users\smero\Desktop\Learning\dataset\images\Sequence05"
model_path = r"C:\Users\smero\Desktop\Learning\saved_result\best_model.pth"
save_folder = r"C:\Users\smero\Desktop\Learning\grad-cam\gradcam_ResNet34_phantom_classification(@seq5)"
os.makedirs(save_folder, exist_ok=True)

# ---- Load model ---- #
args = parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_model = cnn.CNN(args).to(device)
base_model.load_state_dict(torch.load(model_path, map_location=device))
base_model.eval()

# ---- Target layer (ResNet) ---- #
print(base_model.resnet)
target_layers = [base_model.resnet[7][-1]]  # Last conv layer in ResNet34

# # ---- Target layer (AlexNet) ---- #
# print(base_model.features)
# target_layers = [base_model.features[10]]

# ---- Preprocessing ---- #
mean = [0.567, 0.390, 0.388]
std = [0.159, 0.155, 0.168]
input_size = args.input_size if hasattr(args, 'input_size') else 224
transform = A.Compose([
    A.Resize(input_size, input_size),
    A.Normalize(mean=mean, std=std),
    ToTensorV2()
])

# ---- Image file list ---- #
image_files = sorted([
    os.path.join(image_folder, f)
    for f in os.listdir(image_folder)
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
])

# ---- Grad-CAM loop for classification ---- #
for img_path in image_files:
    frame = cv2.imread(img_path)
    if frame is None:
        continue

    transformed = transform(image=frame)
    input_tensor = transformed["image"].unsqueeze(0).to(device)

    # If grayscale, repeat to match 3-channel input
    if input_tensor.shape[1] == 1:
        input_tensor = input_tensor.repeat(1, 3, 1, 1)

    # Convert to RGB for overlay
    rgb_img = frame.astype(np.float32) / 255.0
    rgb_img_resized = cv2.resize(rgb_img, (input_size, input_size))
    name = os.path.splitext(os.path.basename(img_path))[0]

    # ---- Grad-CAM for collision classification ---- #
    # cam_cls = GradCAMPlusPlus(model=CNNWrapperCls(base_model), target_layers=target_layers)
    cam_cls = GradCAM(model=CNNWrapperCls(base_model), target_layers=target_layers)

    cam_output_cls = cam_cls(input_tensor=input_tensor, targets=[ClassificationOutputTarget()])
    cam_image_cls = show_cam_on_image(rgb_img_resized, cam_output_cls[0], use_rgb=True)

    cv2.imwrite(
        os.path.join(save_folder, f"{name}_gradcam_cls.jpg"),
        cv2.cvtColor(cam_image_cls, cv2.COLOR_RGB2BGR)
    )

print(f"[Completed] Grad-CAM results for collision classification have been saved to {save_folder}.")
