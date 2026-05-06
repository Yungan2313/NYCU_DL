import os
import torch
import json
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from diffusers import DDPMScheduler
from tqdm import tqdm
import sys

# 匯入你的模型與資料集類別 (請確認名稱與你的檔案相符)
from train import ConditionalDDPM, sample_pure_conditional
from dataloader import ICLEVRDataset # 替換成你的 Dataset 檔名

# 匯入 Evaluator
sys.path.append('./file')
from file.evaluator import evaluation_model

def evaluate_and_save(model, noise_scheduler, dataloader, evaluator, device, output_dir, dataset_name):
    """
    評估準確率並將個別圖片與 Grid 圖儲存下來
    """
    os.makedirs(os.path.join(output_dir, dataset_name), exist_ok=True)
    eval_transform = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    
    total_acc = 0.0
    all_generated_images = []
    
    # 這裡假設 batch_size 可以一次吃下 32 張測試圖
    for conditions in dataloader:
        conditions = conditions.to(device)
        
        # 1. 生成圖片 (數值範圍 [0, 1])
        generated_images = sample_pure_conditional(model, noise_scheduler, conditions, device)
        all_generated_images.append(generated_images)
        
        # 2. 儲存個別圖片 (1.png, 2.png ...)
        for i, img in enumerate(generated_images):
            # 作業要求依照順序命名，從 1 開始
            save_path = os.path.join(output_dir, dataset_name, f"{i+1}.png")
            save_image(img, save_path)
            
        # 3. 計算準確率 (需轉換為 [-1, 1])
        eval_images = eval_transform(generated_images)
        acc = evaluator.eval(eval_images, conditions)
        total_acc += acc
        
    # 4. 儲存 Grid 圖供報告使用 (8 張一排)
    all_generated_images = torch.cat(all_generated_images, dim=0)
    grid = make_grid(all_generated_images, nrow=8, padding=2)
    save_image(grid, f"{output_dir}/{dataset_name}_grid.png")
    
    avg_acc = total_acc / len(dataloader)
    return avg_acc

@torch.no_grad()
def generate_denoising_process(model, noise_scheduler, target_labels, objects_dict_path, device, save_path):
    """
    生成指定標籤的去噪過程圖
    """
    # 1. 讀取物件字典並轉換為 multi-hot 條件向量
    with open(objects_dict_path, 'r') as f:
        classes = json.load(f)
        
    condition = torch.zeros(1, len(classes)).to(device) # Batch size = 1
    for label in target_labels:
        condition[0, classes[label]] = 1.0
        
    # 2. 初始化雜訊
    x_t = torch.randn((1, 3, 64, 64), device=device)
    noise_scheduler.set_timesteps(1000)
    
    # 決定要在哪些步數擷取影像 (作業要求至少 8 張圖)
    # 我們每隔 125 步擷取一次，加上最後一步 0，共 9 張圖
    capture_steps = set(range(0, 1000, 125))
    capture_steps.add(0) # 確保包含最後清晰的結果
    captured_images = []
    
    for t in tqdm(noise_scheduler.timesteps, desc="Generating Denoising Process"):
        t_batch = torch.full((1,), t, device=device, dtype=torch.long)
        noise_pred = model(x_t, t_batch, condition)
        x_t = noise_scheduler.step(noise_pred, t, x_t).prev_sample
        
        if t.item() in capture_steps:
            # 轉換為 [0, 1] 並存入列表
            img = (x_t.clone() / 2 + 0.5).clamp(0, 1)
            captured_images.append(img)
            
    # 確保順序是從雜訊 (t=999) 到清晰 (t=0)
    captured_images = torch.cat(captured_images, dim=0) # shape: [9, 3, 64, 64]
    
    # 將去噪過程組合成一行 Grid 儲存
    grid = make_grid(captured_images, nrow=len(captured_images), padding=2)
    save_image(grid, save_path)
    print(f"✅ 去噪過程圖已儲存至 {save_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"使用裝置: {device}")
    # --- 1. 初始化模型與排程器 ---
    model = ConditionalDDPM().to(device)
    
    # 替換成你剛剛訓練好的權重路徑
    checkpoint_path = "./checkpoints/v0-150epoch/model_epoch_120.pth" 
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    # 確保推論時的排程器設定與訓練時一致
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
    
    # 初始化 Evaluator
    evaluator = evaluation_model()
    
    # --- 2. 準備測試資料集 ---
    root_dir = './ickevr'
    objects_json = './file/objects.json'
    test_ds = ICLEVRDataset(root_dir, './file/test.json', objects_json, mode='test')
    new_test_ds = ICLEVRDataset(root_dir, './file/new_test.json', objects_json, mode='test')
    
    # 一次將 32 張全部吃進來
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    new_test_loader = DataLoader(new_test_ds, batch_size=32, shuffle=False)
    
    # --- 3. 執行評估與儲存 ---
    output_dir = "./images"
    print("開始測試 test.json ...")
    acc_test = evaluate_and_save(model, noise_scheduler, test_loader, evaluator, device, output_dir, "test")
    
    print("開始測試 new_test.json ...")
    acc_new_test = evaluate_and_save(model, noise_scheduler, new_test_loader, evaluator, device, output_dir, "new_test")
    
    print("\n" + "="*40)
    print(f"🏆 Test Accuracy: {acc_test:.4f}")
    print(f"🏆 New Test Accuracy: {acc_new_test:.4f}")
    print("="*40 + "\n")
    
    # --- 4. 生成作業指定的去噪過程圖 ---
    target_labels = ["red sphere", "cyan cylinder", "cyan cube"]
    denoising_save_path = os.path.join