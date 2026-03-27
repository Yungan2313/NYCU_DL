import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """_summary_
    兩層卷積 => (Conv2d -> ReLu)*2 
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=0),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.double_conv(x)
        
class UNet(nn.Module):
    """_summary_
    downsampling => DoubleConv -> MaxPool2d
    bottleneck => DoubleConv
    upsampling => upConv -> Concatenate -> DoubleConv
    """
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        
        #downsampling
        self.down_conv1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.down_conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.down_conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.down_conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        #bottleneck
        self.bottleneck = DoubleConv(512, 1024)
        
        #upsampling
        self.up_conv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up_doubleconv1 = DoubleConv(1024, 512)
        
        self.up_conv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_doubleconv2 = DoubleConv(512, 256)

        self.up_conv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_doubleconv3 = DoubleConv(256, 128)
       
        self.up_conv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_doubleconv4 = DoubleConv(128, 64)
        
        #output layer
        self.output_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def crop(self, enc_feature, dec_feature):
        """_summary_
        從 encoder 特徵圖中心裁切出與 decoder 特徵圖尺寸相同的區域
        """
        enc_size = enc_feature.size()[2]  # (H, W), 長寬皆等長
        dec_size = dec_feature.size()[2]
        
        delta = (enc_size - dec_size) // 2
        
        return enc_feature[:, :, delta:delta+dec_size, delta:delta+dec_size] # (B, C, H, W)
    
    def forward(self, x):
        
        #downsampling
        x1 = self.down_conv1(x)
        p1 = self.pool1(x1)
        
        x2 = self.down_conv2(p1)
        p2 = self.pool2(x2)
        
        x3 = self.down_conv3(p2)
        p3 = self.pool3(x3)
        
        x4 = self.down_conv4(p3)
        p4 = self.pool4(x4)
        
        #bottleneck
        bn = self.bottleneck(p4)
        
        #upsampling
        up1 = self.up_conv1(bn)
        crop1 = self.crop(x4, up1)
        merge1 = torch.cat([crop1, up1], dim=1)
        dec1 = self.up_doubleconv1(merge1)
        
        up2 = self.up_conv2(dec1)
        crop2 = self.crop(x3, up2)
        merge2 = torch.cat([crop2, up2], dim=1)
        dec2 = self.up_doubleconv2(merge2)
        
        up3 = self.up_conv3(dec2)
        crop3 = self.crop(x2, up3)
        merge3 = torch.cat([crop3, up3], dim=1)
        dec3 = self.up_doubleconv3(merge3)
        
        up4 = self.up_conv4(dec3)
        crop4 = self.crop(x1, up4)
        merge4 = torch.cat([crop4, up4], dim=1)
        dec4 = self.up_doubleconv4(merge4)
        
        #output
        out = self.output_conv(dec4)
        return out
    
if __name__ == '__main__':
    # 測試原始網路尺寸: 輸入 572x572，預期輸出 388x388
    x = torch.randn(1, 3, 572, 572) 
    model = UNet(in_channels=3, out_channels=1)
    
    output = model(x)
    print(f"輸入尺寸: {x.shape}")
    print(f"輸出尺寸: {output.shape}")
        
        