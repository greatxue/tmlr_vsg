import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GCNConv
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.nn import global_mean_pool
import random
import numpy as np

# 设置随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='PROTEINS')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--hidden_dim', type=int, default=64)
args = parser.parse_args()

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
set_seed(args.seed)

# GCN 模型定义
class GCNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(GCNClassifier, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = global_mean_pool(x, batch)  # 图级别池化
        x = self.linear(x)
        return x

# 加载 `PROTEINS` 数据集
dataset = TUDataset(root='./raw_dataset', name=args.dataset)
num_classes = dataset.num_classes
input_dim = dataset.num_features

# 数据集划分
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_data, val_data, test_data = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=args.batch_size)
test_loader = DataLoader(test_data, batch_size=args.batch_size)

# 初始化模型和优化器
model = GCNClassifier(input_dim=input_dim, hidden_dim=args.hidden_dim, num_classes=num_classes).to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

# 训练函数
def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data.x, data.edge_index, data.batch)
        loss = F.cross_entropy(output, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

# 测试函数
@torch.no_grad()
def test(loader):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        output = model(data.x, data.edge_index, data.batch)
        pred = output.argmax(dim=-1)
        correct += (pred == data.y).sum().item()
    return correct / len(loader.dataset)

# 主训练循环
best_val_acc = 0.0
best_test_acc = 0.0

for epoch in range(1, args.epochs + 1):
    loss = train()
    train_acc = test(train_loader)
    val_acc = test(val_loader)
    test_acc = test(test_loader)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_test_acc = test_acc

    print(f"Epoch {epoch}/{args.epochs}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}")

print(f'Best Validation Acc: {best_val_acc:.4f}, Corresponding Test Acc: {best_test_acc:.4f}')