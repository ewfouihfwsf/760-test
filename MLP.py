import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, average_precision_score
import warnings


warnings.filterwarnings("ignore", category=UserWarning)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 21
EPOCHS = 15
BATCH_SIZE = 256
LEARNING_RATE = 1e-3

print(f" 实验开始: 严格 Train/Val/Test 划分的长尾分布 Baseline (Device: {DEVICE})")
print("="*60)



BoW_int = np.load("Extracted_Features/BoW_int.npy").astype(np.float32)
Nor_CH  = np.load("Extracted_Features/Normalized_CH.npy")
Nor_CM  = np.load("Extracted_Features/Normalized_CM55.npy")
Nor_CO  = np.load("Extracted_Features/Normalized_CORR.npy")
Nor_EDH = np.load("Extracted_Features/Normalized_EDH.npy")
Nor_WT  = np.load("Extracted_Features/Normalized_WT.npy")
input_features = np.concatenate([BoW_int, Nor_CH, Nor_CM, Nor_CO, Nor_EDH, Nor_WT], axis=1)

labels = np.load("database_labels.npy").astype(np.float32) 


try:
    Test_BoW = np.load("Extracted_Features_Test/BoW_int.npy").astype(np.float32)
    Test_CH  = np.load("Extracted_Features_Test/Normalized_CH.npy")
    Test_CM  = np.load("Extracted_Features_Test/Normalized_CM55.npy")
    Test_CO  = np.load("Extracted_Features_Test/Normalized_CORR.npy")
    Test_EDH = np.load("Extracted_Features_Test/Normalized_EDH.npy")
    Test_WT  = np.load("Extracted_Features_Test/Normalized_WT.npy")
    

    X_test_raw = np.concatenate([Test_BoW, Test_CH, Test_CM, Test_CO, Test_EDH, Test_WT], axis=1)
    
    Y_test = np.load("test_labels.npy").astype(np.float32)
    
except FileNotFoundError as e:
    print(f" 找不到测试集文件，请检查路径: {e}")
    exit()


scaler = StandardScaler()


input_scaled = scaler.fit_transform(input_features)


X_test_scaled = scaler.transform(X_test_raw)


X_train, X_val, Y_train, Y_val = train_test_split(
    input_scaled, labels, test_size=0.1, random_state=42, shuffle=True
)

class NUS_WIDE(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

train_loader = DataLoader(NUS_WIDE(X_train, Y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(NUS_WIDE(X_val, Y_val), batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(NUS_WIDE(X_test_scaled, Y_test), batch_size=BATCH_SIZE, shuffle=False)

class VanillaMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        return self.net(x)

def train(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    for x, y in dataloader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, threshold=0.5):
    model.eval()
    total_loss = 0
    all_preds, all_targets, all_probs = [], [], []

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item()

            probs = torch.sigmoid(logits)
            all_preds.append((probs > threshold).float().cpu().numpy())
            all_targets.append(y.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    all_probs = np.vstack(all_probs)

    f1_micro = f1_score(all_targets, all_preds, average='micro', zero_division=0)
    f1_macro = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    
    try:
        mAP = average_precision_score(all_targets, all_probs, average='macro')
        per_class_ap = average_precision_score(all_targets, all_probs, average=None)
    except ValueError:
        mAP = 0.0
        per_class_ap = np.zeros(NUM_CLASSES)

    return total_loss / len(dataloader), f1_micro, f1_macro, mAP, per_class_ap

input_dim = input_features.shape[1]
print(f">>> [4/5] 初始化 Vanilla MLP (Input: {input_dim}, Output: {NUM_CLASSES})...")

model = VanillaMLP(input_dim, NUM_CLASSES).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.BCEWithLogitsLoss()

print("\n>>> [5/5] starting training loop...")
for epoch in range(EPOCHS):
    train_loss = train(model, train_loader, optimizer, criterion)
    val_loss, f1_mi, f1_ma, mAP, _ = evaluate(model, val_loader, criterion)
    print(f"Epoch {epoch+1:02d}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | mAP: {mAP:.4f}")



_, test_f1_mi, test_f1_ma, test_mAP, test_per_class_ap = evaluate(model, test_loader, criterion)
class_counts = labels.sum(axis=0)

print("\n Long-tail")
print("mAP:", test_mAP)
print("Macro-F1:", test_f1_ma)

tail_idx = np.argsort(class_counts)[:5]
head_idx = np.argsort(class_counts)[-5:]

print("Head AP:", test_per_class_ap[head_idx].mean())
print("Tail AP:", test_per_class_ap[tail_idx].mean())

print("\n" + "="*55)
print("  FINAL EXAM: THE CURSE OF LONG-TAIL (TEST SET) ")
print("="*55)
print(f" mAP:      {test_mAP:.4f}")
print(f" Micro-F1: {test_f1_mi:.4f}  ")
print(f" Macro-F1: {test_f1_ma:.4f}  ")


valid_ap_indices = [i for i, ap in enumerate(test_per_class_ap) if not np.isnan(ap)]
if len(valid_ap_indices) >= 10:
    sorted_indices = sorted(valid_ap_indices, key=lambda i: test_per_class_ap[i])
    worst_5 = sorted_indices[:5]
    best_5 = sorted_indices[-5:][::-1]

    print("Head Classes:")
    for idx in best_5:
        print(f"   - Class {idx:02d}: AP = {test_per_class_ap[idx]:.4f}")

    print("\nTail Classes:")
    for idx in worst_5:
        print(f"   - Class {idx:02d}: AP = {test_per_class_ap[idx]:.4f}")
print("="*55)