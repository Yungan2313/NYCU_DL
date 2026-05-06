import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel

class ConditionalDDPM(nn.Module):
    def __init__(self, num_classes=24, cross_attention_dim=256):
        super().__init__()
        
        # 1. 條件嵌入網路 (Condition Embedding)
        # 將 24 維的多標籤向量，映射到模型 Cross-Attention 所需的維度
        self.cond_emb = nn.Sequential(
            nn.Linear(num_classes, 128),
            nn.SiLU(),
            nn.Linear(128, cross_attention_dim)
        )
        
        # 2. 呼叫 diffusers 現成的條件式 UNet
        self.unet = UNet2DConditionModel(
            sample_size=64,           # 配合 evaluator 要求的 64x64 解析度
            in_channels=3,            # RGB 輸入
            out_channels=3,           # 預測雜訊輸出
            layers_per_block=2,       # 每個 ResNet 區塊的層數
            block_out_channels=(128, 256, 256, 512), # UNet 各層的特徵維度
            # 定義 Downsample 與 Upsample 的區塊類型 (有 CrossAttn 才能接收條件)
            down_block_types=(
                "CrossAttnDownBlock2D", 
                "CrossAttnDownBlock2D", 
                "CrossAttnDownBlock2D", 
                "DownBlock2D"
            ),
            up_block_types=(
                "UpBlock2D", 
                "CrossAttnUpBlock2D", 
                "CrossAttnUpBlock2D", 
                "CrossAttnUpBlock2D"
            ),
            cross_attention_dim=cross_attention_dim # 必須與我們的 cond_emb 輸出一致
        )

    def forward(self, noisy_images, timesteps, condition_labels):
        """
        noisy_images: 加了雜訊的圖片 [Batch, 3, 64, 64]
        timesteps: 目前的時間步長 [Batch]
        condition_labels: 24維多標籤向量 [Batch, 24]
        """
        # 轉換條件並增加 Sequence 維度 [Batch, 24] -> [Batch, 1, 256]
        encoder_hidden_states = self.cond_emb(condition_labels).unsqueeze(1)
        
        # 預測雜訊
        noise_pred = self.unet(
            noisy_images, 
            timesteps, 
            encoder_hidden_states=encoder_hidden_states
        ).sample
        
        return noise_pred

# ====== 測試模型是否正常運作 ======
if __name__ == "__main__":
    model = ConditionalDDPM().cuda()
    dummy_noisy_images = torch.randn(4, 3, 64, 64).cuda() # Batch=4
    dummy_timesteps = torch.randint(0, 1000, (4,)).cuda()
    dummy_conditions = torch.zeros(4, 24).cuda()
    dummy_conditions[:, [0, 5, 10]] = 1.0 # 模擬 Multi-hot
    
    out = model(dummy_noisy_images, dummy_timesteps, dummy_conditions)
    print(f"Output shape: {out.shape}") # 應該要是 [4, 3, 64, 64]