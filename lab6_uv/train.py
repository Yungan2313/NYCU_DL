import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import sys
from torchvision.utils import make_grid
from diffusers import DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_cosine_schedule_with_warmup
import wandb
from tqdm import tqdm
from dataloader import ICLEVRDataset 
from model import ConditionalDDPM
from file.evaluator import evaluation_model

sys.path.append('./file')

def parse_args():
    parser = argparse.ArgumentParser(description="訓練 Conditional DDPM 模型")
    
    # 訓練基本參數
    parser.add_argument("--epochs", type=int, default=100, help="總訓練回合數")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch Size 大小")
    parser.add_argument("--lr", type=float, default=1e-4, help="初始學習率")
    parser.add_argument("--save_interval", type=int, default=10, help="每隔幾個 epoch 儲存一次模型")
    parser.add_argument("--log_image_interval", type=int, default=5, help="每隔幾個 epoch 紀錄一次生成圖片")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="模型權重儲存路徑")
    parser.add_argument("--eval_interval", type=int, default=10, help="每隔幾個 epoch 用 evaluator 算一次準確率")
    
    # 資料路徑參數
    parser.add_argument("--root_dir", type=str, default="./iclevr", help="圖片根目錄")
    parser.add_argument("--train_json", type=str, default="./file/train.json", help="訓練集 json 路徑")
    parser.add_argument("--test_json", type=str, default="./file/test.json", help="測試集 json 路徑")
    parser.add_argument("--objects_json", type=str, default="./file/objects.json", help="物件字典 json 路徑")
    
    # 模型進階參數
    parser.add_argument("--beta_schedule", type=str, default="squaredcos_cap_v2", choices=["linear", "squaredcos_cap_v2"])
    parser.add_argument("--warmup_steps", type=int, default=500, help="學習率預熱步數")
    
    # WandB 參數
    parser.add_argument("--wandb_project", type=str, default="NYCU_DL_lab6", help="WandB 專案名稱")
    parser.add_argument("--wandb_name", type=str, default="Cosine_AMP_Baseline", help="本次實驗名稱")
    
    return parser.parse_args()

@torch.no_grad()
def sample_pure_conditional(model, noise_scheduler, conditions, device):
    """
    接收測試條件，執行完整的去噪過程，並回傳 [0, 1] 的 RGB 影像張量
    """
    model.eval()
    batch_size = conditions.shape[0]
    
    # 1. 初始化純雜訊圖 x_T
    x_t = torch.randn((batch_size, 3, 64, 64), device=device)
    
    # 2. 設定推論時間步長
    noise_scheduler.set_timesteps(1000)
    
    # 3. 逐步去噪
    for t in tqdm(noise_scheduler.timesteps, desc="Generating Images", leave=False):
        t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
        noise_pred = model(x_t, t_batch, conditions)
        x_t = noise_scheduler.step(noise_pred, t, x_t).prev_sample
        
    # 4. 反標準化：從 [-1, 1] 轉回 [0, 1] 以便 make_grid 和 WandB 顯示
    x_t = (x_t / 2 + 0.5).clamp(0, 1)
    
    return x_t
    
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    run_output_dir = os.path.join(args.output_dir, args.wandb_name)
    os.makedirs(run_output_dir, exist_ok=True)

    # 1. 初始化 WandB
    wandb.login(key="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")
    wandb.init(project=args.wandb_project, name=args.wandb_name, config=vars(args))

    # 2. 資料準備
    full_dataset = ICLEVRDataset(root_dir=args.root_dir, json_path=args.train_json, objects_dict_path=args.objects_json, mode='train')
    train_size = int(0.95 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # --- 新增：準備固定的測試條件 (Fixed Conditions) 用於視覺化追蹤 ---
    test_dataset = ICLEVRDataset(root_dir=args.root_dir, json_path=args.test_json, objects_dict_path=args.objects_json, mode='test')
    # 測試集共 32 筆資料[cite: 5]，我們一次性將它們讀取出來作為固定的條件張量
    fixed_test_conditions = next(iter(DataLoader(test_dataset, batch_size=32, shuffle=False))).to(device)

    # 3. 模型、優化器與排程器
    model = ConditionalDDPM().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule=args.beta_schedule)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.epochs * len(train_loader))
    scaler = torch.amp.GradScaler('cuda')
    
    # 3.5 新增：初始化 Evaluator 實例
    evaluator = evaluation_model()
    # 準備 test.json 和 new_test.json 的 DataLoader
    test_ds = ICLEVRDataset(args.root_dir, './file/test.json', args.objects_json, mode='test')
    new_test_ds = ICLEVRDataset(args.root_dir, './file/new_test.json', args.objects_json, mode='test')
    
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    new_test_loader = DataLoader(new_test_ds, batch_size=32, shuffle=False)
    
    # 4. 訓練迴圈
    best_test_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for images, conditions in pbar:
            images, conditions = images.to(device), conditions.to(device)
            noise = torch.randn_like(images).to(device)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (images.shape[0],), device=device).long()
            noisy_images = noise_scheduler.add_noise(images, noise, timesteps)
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                noise_pred = model(noisy_images, timesteps, conditions)
                loss = F.mse_loss(noise_pred, noise)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            
            train_loss += loss.item()
            current_lr = lr_scheduler.get_last_lr()[0]
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            wandb.log({"step_loss": loss.item(), "learning_rate": current_lr})

        avg_train_loss = train_loss / len(train_loader)
        
        # 5. 驗證與圖片紀錄
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, conditions in val_loader:
                images, conditions = images.to(device), conditions.to(device)
                noise = torch.randn_like(images).to(device)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (images.shape[0],), device=device).long()
                noisy_images = noise_scheduler.add_noise(images, noise, timesteps)
                
                with torch.amp.autocast('cuda'):
                    noise_pred = model(noisy_images, timesteps, conditions)
                    loss = F.mse_loss(noise_pred, noise)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        wandb.log({"epoch": epoch + 1, "epoch_train_loss": avg_train_loss, "epoch_val_loss": avg_val_loss})

        if (epoch + 1) % 10 == 0:
            print(f"\n--- Epoch {epoch+1} : Evaluator test ---")
            
            # 測試 test.json
            acc_test = evaluate_generation(model, noise_scheduler, test_loader, evaluator, device)
            print(f"Test Accuracy: {acc_test:.4f}")
            
            # 測試 new_test.json
            acc_new_test = evaluate_generation(model, noise_scheduler, new_test_loader, evaluator, device)
            print(f"New Test Accuracy: {acc_new_test:.4f}")
            
            # 紀錄到 WandB
            wandb.log({
                "epoch": epoch + 1,
                "Accuracy/test": acc_test,
                "Accuracy/new_test": acc_new_test
            })
            
        
        # --- 新增：視覺化圖片並上傳到 WandB ---
        if (epoch + 1) % args.log_image_interval == 0:
            print(f"Generating images for epoch {epoch+1}...")
            # 使用固定的測試條件 (32張圖) 進行生成
            generated_images = sample_pure_conditional(model, noise_scheduler, fixed_test_conditions, device)
            
            # 使用 torchvision 的 make_grid 排列成網格，作業要求一排 8 張圖片[cite: 1]
            grid = make_grid(generated_images, nrow=8, padding=2, normalize=False)
            
            # 上傳到 WandB
            wandb.log({"Test_Generation_Progress": wandb.Image(grid, caption=f"Epoch {epoch+1} Test.json Results")})

        # 6. 儲存模型
        if (epoch + 1) % args.eval_interval == 0 or (epoch + 1) == args.epochs:
            print(f"\n--- Epoch {epoch+1} : 執行 Evaluator 測試 ---")
            
            # 呼叫你的 evaluate_generation 函數計算 test.json 的準確率
            # (注意：這裡的 evaluate_generation 是我們前面對話中寫的獨立函數)
            acc_test = evaluate_generation(model, noise_scheduler, test_loader, evaluator, device)
            
            wandb.log({"epoch": epoch + 1, "Accuracy/test": acc_test})
            print(f"目前 Test Accuracy: {acc_test:.4f}")

            # 🌟 判斷是否為最佳模型並儲存 🌟
            if acc_test > best_test_acc:
                best_test_acc = acc_test
                best_model_path = os.path.join(run_output_dir, "best_model.pth")
                torch.save(model.state_dict(), best_model_path)
                print(f"new best! best model store in {best_model_path} (Acc: {best_test_acc:.4f})")
        
        if (epoch + 1) % args.save_interval == 0 or (epoch + 1) == args.epochs:
            save_path = os.path.join(run_output_dir, f"model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f" Checkpoint saved to {save_path}")

    wandb.finish()

def evaluate_generation(model, noise_scheduler, test_loader, evaluator, device):
    """
    使用模型生成圖片，並丟入 Evaluator 計算準確率
    """
    model.eval()
    
    # 準備 Evaluator 專用的標準化轉換
    # 將 [0, 1] 的影像轉換為 [-1, 1]
    eval_transform = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    
    total_acc = 0.0
    total_batches = 0
    
    with torch.no_grad():
        # 通常 test_loader 只有一個 Batch (包含 test.json 的 32 張)
        for conditions in test_loader:
            conditions = conditions.to(device)
            
            # 1. 呼叫之前的生成函數 (回傳的是 [0, 1] 的影像張量)
            generated_images_0_to_1 = sample_pure_conditional(model, noise_scheduler, conditions, device)
            
            # 2. 為 Evaluator 進行標準化 (轉回 [-1, 1])
            eval_images = eval_transform(generated_images_0_to_1)
            
            # 3. 丟入 Evaluator 計算準確率
            # 注意：evaluator 預期的是 normalized images 和 multi-hot labels
            acc = evaluator.eval(eval_images, conditions)
            
            total_acc += acc
            total_batches += 1
            
    avg_acc = total_acc / total_batches
    return avg_acc

if __name__ == "__main__":
    args = parse_args()
    train(args)
    
    
    # uv run train.py --epochs 5 --batch_size 16 --wandb_name "H200_Quick_Test"
    # uv run train.py --epochs 150 --batch_size 64 --lr 2e-4 --wandb_name "v0-150epoch" --save_interval 20 --log_image_interval 20
    # uv run train.py --epochs 150 --batch_size 64 --lr 2e-4 --wandb_name "v0-150epoch-linear" --save_interval 10 --log_image_interval 20 --beta_schedule linear
    
    # uv run hf upload yuminyu/NCYU_DL_lab6 ./results/images/ /generated_images/