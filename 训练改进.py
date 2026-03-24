import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, average_precision_score

# ==========================================
# 1. 自动推断设备 (GPU/CPU)
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

# ==========================================
# 2. 数据加载与预处理 (使用清洗后的 TC-21)
# ==========================================
print(">>> Loading extracted features and TC-21 labels...")

# 加载特征 (X)
try:
    BoW_int = np.load("Extracted_Features/BoW_int.npy").astype(np.float32)
    Nor_CH = np.load("Extracted_Features/Normalized_CH.npy")
    Nor_CM = np.load("Extracted_Features/Normalized_CM55.npy")
    Nor_CO = np.load("Extracted_Features/Normalized_CORR.npy")
    Nor_EDH = np.load("Extracted_Features/Normalized_EDH.npy")
    Nor_WT = np.load("Extracted_Features/Normalized_WT.npy")
    
    # 拼接所有训练集特征
    input_features = np.concatenate([BoW_int, Nor_CH, Nor_CM, Nor_CO, Nor_EDH, Nor_WT], axis=1)
    
    # 加载标签 (Y)
    labels = np.load("database_labels.npy").astype(np.float32)
    
except FileNotFoundError as e:
    print(f"❌ Error loading files: {e}. Please check paths.")
    exit()

# ⚠️ 关键修改点 1：标准化 (StandardScaler)
print(">>> Applying StandardScaler...")
scaler = StandardScaler()
input_scaled = scaler.fit_transform(input_features)

# ⚠️ 关键修改点 2：我们不用 train_test_split，因为官方已经给好了 test_img.txt
# 目前我们先把这 193734 当作 Train+Val (取 90% 做训练，10% 做验证)
# 测试集 (2100张) 应该在另一个独立的文件里，这里先用 Validation 验证模型收敛
from sklearn.model_selection import train_test_split
X_train, X_val, Y_train, Y_val = train_test_split(
    input_scaled, labels,
    test_size=0.1, # 留 10% 做验证集
    random_state=42,
    shuffle=True
)

class NUS_WIDE(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

train_dataset = NUS_WIDE(X_train, Y_train)
val_dataset   = NUS_WIDE(X_val, Y_val)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=0)

print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")


# ==========================================
# 3. 模型定义 (修改为 21 类)
# ==========================================
class MultiLabelModel(nn.Module):
    # ⚠️ 关键修改点 3：num_labels 从 81 改为 21
    def __init__(self, input_dim, hidden_dim=256, embed_dim=128, num_labels=21):
        super().__init__()
        self.bn_input = nn.BatchNorm1d(input_dim)
        
        # MLP encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(512, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.5)
        )

        # Classifier branch
        self.classifier = nn.Linear(hidden_dim, num_labels)

        # Label embedding branch
        self.feature_proj = nn.Linear(hidden_dim, embed_dim)
        self.label_embedding = nn.Embedding(num_labels, embed_dim)

    def forward(self, x):
        x = self.bn_input(x)
        h = self.encoder(x)

        # Branch 1: classification
        logits_cls = self.classifier(h)

        # Branch 2: embedding
        feat_embed = F.normalize(self.feature_proj(h), dim=-1)
        label_embed = F.normalize(self.label_embedding.weight, dim=-1)
        logits_embed = torch.matmul(feat_embed, label_embed.t())

        # Combine
        logits = logits_cls + logits_embed
        return logits, logits_embed


# ==========================================
# 4. 损失函数定义
# ==========================================
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce)
        loss = self.alpha * (1 - pt) ** self.gamma * bce
        return loss.mean()

def contrastive_loss(logits, targets, temperature=0.1):
    logits = logits / temperature
    exp_logits = torch.exp(logits)
    denom = exp_logits.sum(dim=1, keepdim=True)
    num = (exp_logits * targets).sum(dim=1, keepdim=True)
    loss = -torch.log(num / (denom + 1e-8) + 1e-8)
    return loss.mean()


# ==========================================
# 5. 训练与评估函数
# ==========================================
def train(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    focal = FocalLoss()

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        logits, logits_embed = model(x)
        
        # ⚠️ 关键修改点 4：防止 contrastive_loss 因为全0标签行报错
        # 在多标签数据中，极少部分图片可能没有任何这21类标签，对比学习会除以0
        # 添加一个小掩码过滤掉全 0 的行
        valid_mask = y.sum(dim=1) > 0
        loss_cls = focal(logits, y)
        
        if valid_mask.sum() > 0:
            loss_con = contrastive_loss(logits_embed[valid_mask], y[valid_mask])
            loss = loss_cls + 0.1 * loss_con
        else:
            loss = loss_cls

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)

def evaluate(model, dataloader, device, threshold=0.5):
    model.eval()
    total_loss = 0
    focal = FocalLoss()
    all_preds, all_targets, all_probs = [], [], []

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits, logits_embed = model(x)
            
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float()

            valid_mask = y.sum(dim=1) > 0
            loss_cls = focal(logits, y)
            if valid_mask.sum() > 0:
                loss_con = contrastive_loss(logits_embed[valid_mask], y[valid_mask])
                loss = loss_cls + 0.1 * loss_con
            else:
                loss = loss_cls

            total_loss += loss.item()
            
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    all_probs = np.vstack(all_probs)

    # ⚠️ 关键修改点 5：避免 zero_division 警告
    acc = (all_preds == all_targets).all(axis=1).mean()
    f1_micro = f1_score(all_targets, all_preds, average='micro', zero_division=0)
    f1_macro = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    
    # 如果某一类在 batch 里全 0，mAP 会报错，加个判断
    try:
        mAP = average_precision_score(all_targets, all_probs, average='macro')
    except ValueError:
        mAP = 0.0

    return total_loss / len(dataloader), acc, f1_micro, f1_macro, mAP


# ==========================================
# 6. 主循环开始训练
# ==========================================
# 动态获取 input_dim (根据拼接后的特征实际列数)
if __name__ == '__main__':
    actual_input_dim = input_features.shape[1]
    print(f">>> Initializing model with input_dim={actual_input_dim}, num_labels=21...")

    model = MultiLabelModel(input_dim=actual_input_dim, num_labels=21).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3) # 加了一点权重衰减防止过拟合
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    best_val_loss = float('inf')

    print("\n🚀 Starting Training Loop...")
    for epoch in range(20):
        train_loss = train(model, train_loader, optimizer, device)
        val_loss, acc, f1_micro, f1_macro, mAP = evaluate(model, val_loader, device)
        scheduler.step(val_loss)
        print(f"Epoch {epoch+1:02d}/{20} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"mAP: {mAP:.4f} | "
            f"F1-mi: {f1_micro:.4f} | "
            f"F1-ma: {f1_macro:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("   🌟 Best model saved!")

    print("\n🎉 Training Complete!")