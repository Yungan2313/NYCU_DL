import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime

from src.oxford_pet import OxfordPetDataset
from src.models.unet import UNet
from src.models.resnet34_unet import ResNet34_UNet

class InferenceConfig:
    model_name = "unet"
    # model_name = "resnet34_unet"
    data_dir = "dataset/oxford-iiit-pet"
    split_dir = "dataset/oxford-iiit-pet/unet"
    # split_dir = "dataset/oxford-iiit-pet/res_unet"
    img_size = 572
    batch_size = 1
    dice_threshold = 0.5
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model_path = "./log/0327_0121/best_unet.pth"
    
    output_dir = "./results"
    
def rle_encode(mask):
    pixels = mask.flatten(order='F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)
    
def unnormalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return torch.clamp(tensor, 0, 1)

def main():
    cfg = InferenceConfig()
    
    timestamp = datetime.now().strftime("%m%d_%H%M")
    output_dir = f"./results/{cfg.model_name}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    model_info = {
        "model_name": cfg.model_name,
        "inference_time": timestamp,
        "img_size": cfg.img_size
    }
    with open(os.path.join(output_dir, "model_info.json"), "w") as f:
        json.dump(model_info, f, indent=4)
    
    if cfg.model_name == "unet":
        model = UNet(in_channels=3, out_channels=1).to(cfg.device)
    elif cfg.model_name == "resnet34_unet":
        model = ResNet34_UNet(in_channels=3, out_channels=1).to(cfg.device)
        
    model.load_state_dict(torch.load(cfg.model_path, map_location=cfg.device))
    model.eval()
    test_datset = OxfordPetDataset(
        cfg.data_dir, cfg.split_dir, mode="test",
        img_size=(cfg.img_size, cfg.img_size)
    )
    dataset_mean, dataset_std = test_datset.mean, test_datset.std
    test_loader = DataLoader(test_datset, batch_size=cfg.batch_size, shuffle=False)
    
    submission_data = []
    saved_img_count = 0
    print("start inference...")
    with torch.no_grad():
        for images, _, img_names in tqdm(test_loader, desc="Inference"):
            images = images.to(cfg.device)
            img_name = img_names[0]
            
            outputs = model(images)
            _, _, H_in, W_in = images.shape
            _, _, H_out, W_out = outputs.shape
            
            pad_y = (H_in - H_out) // 2
            pad_x = (W_in - W_out) // 2
            
            if pad_y > 0 or pad_x > 0:
                outputs = F.pad(outputs, (pad_x, pad_x, pad_y, pad_y), mode='constant', value=0)
                images_padded = F.pad(images, (pad_x, pad_x, pad_y, pad_y), mode='reflect')
                outputs = model(images_padded)
            
            preds_for_plot = torch.sigmoid(outputs) > cfg.dice_threshold
            pred_mask_for_plot = preds_for_plot[0, 0].cpu().numpy().astype(np.uint8)
            orig_img_path = os.path.join(cfg.data_dir, "images", f"{img_name}.jpg")
            orig_w, orig_h = Image.open(orig_img_path).size
            outputs_resized = F.interpolate(outputs, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
            preds_orig = torch.sigmoid(outputs_resized) > cfg.dice_threshold
            pred_mask_orig = preds_orig[0, 0].cpu().numpy().astype(np.uint8)
            
            rle_str = rle_encode(pred_mask_orig)
            submission_data.append({"image_id": img_name, "encoded_mask": rle_str})
            if saved_img_count < 15:
                img_cpu = images[0].cpu()
                img_unnorm = unnormalize(img_cpu.clone(), dataset_mean, dataset_std)
                img_np = np.transpose(img_unnorm.numpy(), (1, 2, 0)) # (C, H, W) -> (H, W, C)
                
                # === 修改處：改為 1x2 的畫布 ===
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                
                # 左圖：原圖
                axes[0].imshow(img_np)
                axes[0].set_title("Original Image")
                axes[0].axis('off')
                
                # 右圖：模型預測疊加
                axes[1].imshow(img_np)
                axes[1].imshow(pred_mask_for_plot, cmap='jet', alpha=0.5)
                axes[1].set_title("Model Prediction")
                axes[1].axis('off')
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"{img_name}_pred.png"))
                plt.close()
                
                saved_img_count += 1
            
        csv_path = f"submission_{cfg.model_name}.csv"
        df = pd.DataFrame(submission_data)
        df.to_csv(csv_path, index=False)
        print(f"Submission CSV saved to: {csv_path}")
        print(f"Predicted masks saved to: {output_dir}")
        
if __name__ == "__main__":
    main()