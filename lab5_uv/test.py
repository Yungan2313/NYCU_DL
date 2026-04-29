import torch

def test_pytorch_gpu():
    print("===== PyTorch GPU 測試 =====")
    
    # 1. 檢查 CUDA 是否可用
    is_available = torch.cuda.is_available()
    print(f"CUDA 是否可用: {is_available}")

    if is_available:
        # 2. 取得 GPU 數量
        device_count = torch.cuda.device_count()
        print(f"偵測到的 GPU 數量: {device_count}")

        # 3. 取得目前 GPU 的名稱
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        print(f"目前使用的 GPU: {device_name}")

        # 4. 進行一個簡單的運算測試 (張量搬運至 GPU)
        print("\n正在執行運算測試...")
        try:
            # 建立兩個隨機矩陣並移至 GPU
            a = torch.rand(3000, 3000).to("cuda")
            b = torch.rand(3000, 3000).to("cuda")
            
            # 矩陣相乘
            c = torch.matmul(a, b)
            
            print("運算測試成功！矩陣已在 GPU 上完成計算。")
            print(f"張量所在的裝置: {c.device}")
        except Exception as e:
            print(f"運算測試失敗: {e}")
    else:
        print("\n[警告] 未偵測到 CUDA。請檢查 Nvidia 驅動程式或 PyTorch 版本是否正確（是否安裝了 CPU 版本）。")

if __name__ == "__main__":
    test_pytorch_gpu()