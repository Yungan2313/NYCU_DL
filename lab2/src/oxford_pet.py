import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from tqdm import tqdm

class OxfordPetDataset(Dataset):
    def __init__(self, data_dir, split_dir, mode, img_size=(256,256)):
        """_summary_
        Args:
            data_dirl (_type_): 資料集路徑(dataset/oxford-pet)
            split_dir (_type_): 資料切割路徑(dataset/oxford-pet/unet)
            mode (_type_): 資料模式 (train, val, test)
            img_size (tuple, optional): 預處理後的圖片大小
        """
        self.data_dir = data_dir
        self.split_dir = split_dir
        self.img_size = img_size
        self.mode = mode
        self.img_size = img_size

        # 讀取資料切割檔案
        if self.mode == "train":
            list_file_path = os.path.join(split_dir, "train.txt")
        elif self.mode == "val":
            list_file_path = os.path.join(split_dir, "val.txt")
        elif self.mode == "test":
            if "res_unet" in split_dir:
                list_file_path = os.path.join(split_dir, "test_res_unet.txt")
            else:
                list_file_path = os.path.join(split_dir, "test_unet.txt")
        else:
            raise ValueError("Invalid mode. Choose from 'train', 'val', or 'test'.")
        # 讀取影像名稱列表
        self.img_names = []
        with open(list_file_path, 'r') as f:
            for line in f:
                line = line.strip() # 移除行尾的換行符號
                if line:
                    self.img_names.append(line.split(' ')[0]) # 取出圖片名稱
                    
        # 讀取該train.txt計算mean, std
        train_list_file = os.path.join(self.split_dir, 'train.txt')
        self.mean, self.std = self._calculate_stats(train_list_file)
        print(f"Mean: {self.mean}, Std: {self.std}")
    
    def _calculate_stats(self, train_list_file):
        """_summary_
        計算訓練集的mean, std
        """
        train_img_names = []
        with open(train_list_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    train_img_names.append(line.split(' ')[0])
                    
        pixel_sum = torch.tensor([0.0, 0.0, 0.0]) #pixel總和
        pixel_sum_sq = torch.tensor([0.0, 0.0, 0.0]) #pixel平方和
        num_pixels = 0
        
        for img_name in tqdm(train_img_names, desc=f"Calculating Stats for {self.mode}"):
            img_path = os.path.join(self.data_dir, "images", f"{img_name}.jpg")
            if not os.path.exists(img_path):
                continue
            
            image = Image.open(img_path).convert("RGB")
            image = TF.resize(image, self.img_size, interpolation=Image.BILINEAR)
            image = TF.to_tensor(image)
            
            pixel_sum += image.sum(dim=[1, 2])
            pixel_sum_sq += (image ** 2).sum(dim=[1, 2])
            num_pixels += image.shape[1] * image.shape[2]
            
        mean = (pixel_sum / num_pixels).tolist()
        std = torch.sqrt((pixel_sum_sq / num_pixels) - torch.tensor(mean) ** 2).tolist()
        return mean, std
        
    def __len__(self):
        return len(self.img_names)
    
    def preprocess(self, image, mask):
        """_summary_
        對資料做預處理
        確保image和mask的尺寸相同
        """
        # 調整圖片大小
        image = TF.resize(image, self.img_size, interpolation=Image.BILINEAR) #縮放時 => 加權平均得到顏色
        mask = TF.resize(mask, self.img_size, interpolation=Image.NEAREST) #縮放決定顏色 => 避免出現 0 是背景 1 是貓時產生 0.5(沒有意義)
        
        # 將 PIL Image 轉換為 PyTorch Tensor
        image = TF.to_tensor(image)
        
        # 將 Mask 轉換為 numpy array => 標籤轉換
        mask_np = np.array(mask, dtype=np.int64)
        
        # 建立binary_mask(全為0)
        binary_mask = np.zeros_like(mask_np, dtype=np.int64)
        
        binary_mask[mask_np == 1] = 1 # 將貓的像素位置設為1
        binary_mask = torch.from_numpy(binary_mask).unsqueeze(0).float() # (H,W) => (1,H,W)
        
        # 影像正規化
        image = TF.normalize(image, mean=self.mean, std=self.std)
            
        return image, binary_mask
    
    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        
        # 組裝檔案路徑
        img_path = os.path.join(self.data_dir, "images", f"{img_name}.jpg")
        mask_path = os.path.join(self.data_dir, "annotations", "trimaps", f"{img_name}.png")
        
        # 讀取影像
        image = Image.open(img_path).convert("RGB")
        
        if self.mode == "test":
            mask = Image.new("L", image.size, 0)
        else:
            mask = Image.open(mask_path)
        
        # 預處理影像和 Mask
        image, binary_mask = self.preprocess(image, mask)
        
        return image, binary_mask, img_name
    
if __name__ == "__main__":
    #下方為測試資料讀取情況，由gemini生成
    TEST_DATA_DIR = "dataset/oxford-iiit-pet/"
    TEST_SPLIT_DIR = "dataset/oxford-iiit-pet/unet"

    try:
        print("=== 測試開始：初始化 OxfordPetDataset ===")
        # 測試 Train 模式
        train_dataset = OxfordPetDataset(
            data_dir=TEST_DATA_DIR, 
            split_dir=TEST_SPLIT_DIR, 
            mode="train"
        )
        print(f"✅ 成功載入 Train Dataset，總資料筆數: {len(train_dataset)}")
        print(f"✅ 計算得到的 Mean: {train_dataset.mean}")
        print(f"✅ 計算得到的 Std:  {train_dataset.std}")
        print("-" * 40)
        
        # 搭配 DataLoader 測試讀取行為
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        
        # 抽取一個 Batch 來觀察
        images, masks = next(iter(train_loader))
        
        print("=== 測試 Batch 讀取 ===")
        print(f"✅ 影像 Tensor 尺寸: {images.shape}")  # 預期: (4, 3, 256, 256)
        print(f"✅ Mask Tensor 尺寸: {masks.shape}")   # 預期: (4, 1, 256, 256)
        
        # 驗證 Mask 是否只有 0 和 1
        unique_values = torch.unique(masks)
        print(f"✅ Mask 內的唯一值: {unique_values.tolist()} (預期應只包含 0 和 1)")
        
        if set(unique_values.tolist()).issubset({0.0, 1.0}):
            print("🎉 測試通過！資料集預處理與正規化運作正常。")
        else:
            print("⚠️ 警告：Mask 中包含 0 和 1 以外的數值，請檢查標籤轉換邏輯。")

    except FileNotFoundError as e:
        print(f"❌ 找不到檔案或目錄: {e}")
        print("💡 請確認您的終端機執行路徑，或是 dataset 內部是否已經放好圖片與 txt 檔。")
    except Exception as e:
        print(f"❌ 發生未預期的錯誤: {e}")