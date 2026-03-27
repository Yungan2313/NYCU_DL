import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json
import time
import random
from datetime import datetime
from dataclasses import dataclass, asdict
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from src.oxford_pet import OxfordPetDataset
from src.models.unet import UNet
from src.models.resnet34_unet import ResNet34_UNet


#config
@dataclass
class TrainConfig:
    # model_name: str = "unet" 
    model_name: str = "resnet34_unet"
    
    data_dir: str = "dataset/oxford-iiit-pet"
    # split_dir: str = "dataset/oxford-iiit-pet/unet"
    split_dir: str = "dataset/oxford-iiit-pet/res_unet"
    img_size: int = 572
    
    #train
    epochs: int = 20
    batch_size: int = 2
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    max_grad_norm: float = 1.0 
    dice_threshold: float = 0.5
    
    #system
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 1
    save_model_dir: str = "./saved_models"
    log_base_dir: str = "./log"

def calculate_dice_score(pred_logits, targets, smooth=1e-5, threshold=0.5):
    """_summary_
    計算 Dice Score。
    公式: 2 * intersection / (pred_size + gt_size)
    """
    #將Logits轉為機率，再分為0或1
    preds = torch.sigmoid(pred_logits) > threshold
    preds = preds.float()
    
    #計算交集與聯集
    preds = preds.reshape(-1)
    targets = targets.reshape(-1)
    
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum()
    
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.item()

def unnormalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return torch.clamp(tensor, 0, 1)

#visualization
def save_metrics_plot(train_metrics, val_metrics, title, ylabel, save_path):
    """繪製並儲存 Loss 或 Accuracy(Dice) 曲線圖"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_metrics, label=f'Train {ylabel}', color='blue')
    plt.plot(val_metrics, label=f'Validation {ylabel}', color='orange')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def save_comparison_grid(model, val_loader, cfg, save_path, dataset_mean, dataset_std):
    """隨機挑選 6 張圖片，繪製 Original / Ground Truth / Prediction 對比圖"""
    model.eval()
    images, masks, _ = next(iter(val_loader)) # 抓取一個 Batch
    images = images.to(cfg.device)
    masks = masks.to(cfg.device)
    
    with torch.no_grad():
        preds_logits = model(images)
        preds = torch.sigmoid(preds_logits) > 0.5 # 轉為二值化 Mask
        preds = preds.float()
    
    _, _, H_out, W_out = preds.shape
    _, _, H_in, W_in = images.shape
    if H_out < H_in or W_out < W_in:
            diff_y = (H_in - H_out) // 2
            diff_x = (W_in - W_out) // 2
            pad_left = diff_x
            pad_right = W_in - W_out - diff_x
            pad_top = diff_y
            pad_bottom = H_in - H_out - diff_y
            
            preds = F.pad(preds, (pad_left, pad_right, pad_top, pad_bottom), value=0)
    
    # 隨機選擇最多 6 張圖的 Index
    num_images = min(6, images.size(0))
    indices = random.sample(range(images.size(0)), num_images)
    
    fig, axes = plt.subplots(num_images, 3, figsize=(12, 4 * num_images))
    if num_images == 1: axes = [axes]
    
    for row_idx, img_idx in enumerate(indices):
        # 取得圖片並還原正規化
        img = images[img_idx].cpu()
        img = unnormalize(img.clone(), dataset_mean, dataset_std)
        img_np = np.transpose(img.numpy(), (1, 2, 0)) # (C, H, W) -> (H, W, C)
        
        # 取得 Ground Truth 與 Prediction
        gt_mask = masks[img_idx].cpu().squeeze().numpy()
        pred_mask = preds[img_idx].cpu().squeeze().numpy()
        
        # 繪圖
        axes[row_idx][0].imshow(img_np)
        axes[row_idx][0].set_title("Original Image")
        axes[row_idx][0].axis('off')
        
        axes[row_idx][1].imshow(img_np) # 疊加顯示比較清楚
        axes[row_idx][1].imshow(gt_mask, cmap='jet', alpha=0.5)
        axes[row_idx][1].set_title("Ground Truth")
        axes[row_idx][1].axis('off')
        
        axes[row_idx][2].imshow(img_np)
        axes[row_idx][2].imshow(pred_mask, cmap='jet', alpha=0.5)
        axes[row_idx][2].set_title("Model Prediction")
        axes[row_idx][2].axis('off')
        
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    cfg = TrainConfig()
    
    #directory setup
    timestamp = datetime.now().strftime("%m%d_%H%M")
    log_dir = os.path.join(cfg.log_base_dir, timestamp)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(cfg.save_model_dir, exist_ok=True)
    
    with open(os.path.join(log_dir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=4)
        
    print("train start\n")
    
    train_dataset = OxfordPetDataset(cfg.data_dir, cfg.split_dir, mode="train", img_size=(cfg.img_size, cfg.img_size))
    train_mean, train_std = train_dataset.mean, train_dataset.std 
    val_dataset = OxfordPetDataset(cfg.data_dir, cfg.split_dir, mode="val", img_size=(cfg.img_size, cfg.img_size))
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    if cfg.model_name == "unet":
        model = UNet(in_channels=3, out_channels=1).to(cfg.device)
    elif cfg.model_name == "resnet34_unet":
        model = ResNet34_UNet(in_channels=3, out_channels=1).to(cfg.device)
        
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    
    history = {'train_loss': [], 'train_dice': [], 'val_loss': [], 'val_dice': []}
    best_val_dice = 0.0
    
    # --- 開始訓練 ---
    for epoch in range(cfg.epochs):
        model.train()
        train_loss, train_dice = 0.0, 0.0
        
        for images, masks, img_names in tqdm(train_loader, desc="Training"):
            images, masks = images.to(cfg.device), masks.to(cfg.device)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            _, _, H_out, W_out = outputs.shape 
            _, _, H_in, W_in = masks.shape
            diff_y = (H_in - H_out) // 2
            diff_x = (W_in - W_out) // 2
            masks_cropped = masks[:, :, diff_y : diff_y + H_out, diff_x : diff_x + W_out]
            
            loss = criterion(outputs, masks_cropped)
            loss.backward()
            
            nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            train_dice += calculate_dice_score(outputs, masks_cropped, threshold=cfg.dice_threshold) * images.size(0)
            
        train_loss /= len(train_dataset)
        train_dice /= len(train_dataset)
        
        #valid
        model.eval()
        val_loss, val_dice = 0.0, 0.0
        with torch.no_grad():
            for images, masks, img_names in tqdm(val_loader, desc="Validation"):
                images, masks = images.to(cfg.device), masks.to(cfg.device)
                outputs = model(images)
                
                _, _, H_out, W_out = outputs.shape 
                _, _, H_in, W_in = masks.shape
                diff_y = (H_in - H_out) // 2
                diff_x = (W_in - W_out) // 2
                masks_cropped = masks[:, :, diff_y : diff_y + H_out, diff_x : diff_x + W_out]
                
                loss = criterion(outputs, masks_cropped)
                
                val_loss += loss.item() * images.size(0)
                val_dice += calculate_dice_score(outputs, masks_cropped, threshold=cfg.dice_threshold) * images.size(0)
                
        val_loss /= len(val_dataset)
        val_dice /= len(val_dataset)
        
        #record history
        history['train_loss'].append(train_loss)
        history['train_dice'].append(train_dice)
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_dice)
        
        print(f"Epoch [{epoch+1:02d}/{cfg.epochs:02d}] "
              f"Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f} || "
              f"Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}")
        
        #best model save
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            log_save_path = os.path.join(log_dir, f"best_{cfg.model_name}.pth")
            torch.save(model.state_dict(), log_save_path)
            save_path = os.path.join(cfg.save_model_dir, f"best_{cfg.model_name}_{timestamp}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved with Val Dice: {best_val_dice:.4f}")

    #graph
    save_metrics_plot(history['train_loss'], history['val_loss'], 'Loss Curve', 'Loss', os.path.join(log_dir, 'loss.png'))
    save_metrics_plot(history['train_dice'], history['val_dice'], 'Dice Score Curve', 'Dice Score', os.path.join(log_dir, 'acc.png'))
    save_comparison_grid(model, val_loader, cfg, os.path.join(log_dir, 'predictions_grid.png'), train_mean, train_std)
    
    print("Training complete")

if __name__ == '__main__':
    main()