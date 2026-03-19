import torch
import torch.nn as nn

# 1. 图像特征编码器 (Image Encoder)
# 把原始的 48 维图片特征，变成 64 维
class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(48, 128),  # 升维提取一下特征
            nn.ReLU(),           # 激活函数
            nn.Linear(128, 64)   # 降维到 64 维
        )

    def forward(self, x):
        return self.net(x)

# 2. 文本特征编码器 (Text Encoder)
# 把原始的 40 维文本特征，也变成 64 维
class TextEncoder(nn.Module):
    def __init__(self):
        super(TextEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(40, 128),  
            nn.ReLU(),
            nn.Linear(128, 64)   # 同样降维到 64 维，这样图和文就在同一个世界了！
        )

    def forward(self, x):
        return self.net(x)

# --- 测试一下网络是不是通的 ---
if __name__ == "__main__":
    # 假装我们拿到了 5 个样本的图片和文本特征
    dummy_img = torch.randn(5, 48)  # 5个样本，48维
    dummy_txt = torch.randn(5, 40)  # 5个样本，40维
    
    img_model = ImageEncoder()
    txt_model = TextEncoder()
    
    out_img = img_model(dummy_img)
    out_txt = txt_model(dummy_txt)
    
    print("图像通过网络后的形状:", out_img.shape) # 应该是 [5, 64]
    print("文本通过网络后的形状:", out_txt.shape) # 应该是 [5, 64]