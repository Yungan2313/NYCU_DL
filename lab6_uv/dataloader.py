import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

class ICLEVRDataset(Dataset):
    def __init__(self, root_dir, json_path, objects_dict_path, mode='train'):
        """
        dataset
        :param root_dir: root directory for the images ('./iclevr')
        :param json_path: path to the dataset json file (train.json or test.json)
        :param objects_dict_path: path to the objects.json file
        :param mode: 'train' or 'test'
        """
        self.root_dir = root_dir
        self.mode = mode

        # 讀取物件對應字典，共 24 種類別
        with open(objects_dict_path, 'r') as f:
            self.classes = json.load(f)
        self.num_classes = len(self.classes) 

        # 讀取資料 json
        with open(json_path, 'r') as f:
            self.data = json.load(f)

        # 根據 mode 處理資料結構
        if self.mode == 'train':
            self.img_names = list(self.data.keys())
            self.labels = list(self.data.values())
        elif self.mode == 'test':
            # 測試集的格式為 [["物件1"], ["物件2", "物件3"]]
            self.labels = self.data
        else:
            raise ValueError("mode 必須是 'train' 或 'test'")

        # 依照 evaluator 的要求設定 Transforms
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.labels)

    def get_multi_hot(self, label_list):
        """將字串標籤列表轉換為 24 維的 Multi-hot vector"""
        multi_hot = torch.zeros(self.num_classes)
        for obj in label_list:
            class_idx = self.classes[obj]
            multi_hot[class_idx] = 1.0
        return multi_hot

    def __getitem__(self, idx):
        # 取得標籤列表並轉換為 multi-hot tensor[cite: 2]
        label_list = self.labels[idx]
        condition = self.get_multi_hot(label_list)

        if self.mode == 'train':
            # 訓練階段：讀取圖片並進行 transforms
            img_name = self.img_names[idx]
            img_path = os.path.join(self.root_dir, img_name)
            
            # 確保圖片為 RGB 格式
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
            
            return image, condition
        else:
            # 測試階段：只需回傳條件向量即可
            return condition


# ==========================================
# 使用範例與 DataLoader 建立封裝函數
# ==========================================
def get_dataloader(root_dir, json_path, objects_dict_path, mode='train', batch_size=64, num_workers=4):
    dataset = ICLEVRDataset(
        root_dir=root_dir, 
        json_path=json_path, 
        objects_dict_path=objects_dict_path, 
        mode=mode
    )
    
    # 訓練時打亂順序，測試時則保持 json 中的順序
    shuffle = True if mode == 'train' else False
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers,
        pin_memory=True # 加速資料傳輸到 GPU
    )
    return dataloader

if __name__ == '__main__':
    # 簡單的測試程式碼 (請確保檔案路徑正確)
    train_loader = get_dataloader('./iclevr', './file/train.json', './file/objects.json', mode='train', batch_size=16)
    for images, conditions in tqdm(train_loader, desc="Training"):
        print(f"Images shape: {images.shape}")       # 預期: torch.Size([16, 3, 64, 64])
        print(f"Conditions shape: {conditions.shape}") # 預期: torch.Size([16, 24])
        break
    pass