import argparse
import os
import os.path as osp
import pandas as pd
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch.utils.data import Dataset, random_split
from torchvision import transforms
from torchvision.models import convnext_small
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from PIL import Image
import random
import numpy as np
torch.cuda.empty_cache()

# 设置随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)

# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='PROTEINS')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--image_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--normalize', action='store_true', help='Apply Normalization')
args = parser.parse_args()

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
set_seed(args.seed)

# 数据集类
class MultiModalDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform if transform is None else transforms.ToTensor()

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # 加载图像
        img_path = osp.join(self.root_dir, self.annotations.iloc[idx, 0])
        
        #print(f"[DEBUG] Image path: {img_path}")
        
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        #print(f"[DEBUG] Loaded image: {img_path}, Size: {image.size}")

        # 加载图数据结构
        graph_path = self.annotations.iloc[idx, 1]
        
        #print(f"[DEBUG] Graph path: {graph_path}")
        
        with open(graph_path, 'rb') as f:
            graph_data = pickle.load(f)
        
        #print(f"[DEBUG] Loaded graph data: {graph_path}, Graph nodes: {graph_data.num_nodes}, Graph edges: {graph_data.edge_index.size()}")

        # 获取标签
        label = int(self.annotations.iloc[idx, 2])
        return image, graph_data, label

# 图像数据变换
transform = transforms.Compose([
    transforms.Resize((args.image_size, args.image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) if args.normalize else None
])

# 加载数据集
dataset = MultiModalDataset(csv_file=f'./modality/{args.dataset}/dataset.csv', root_dir='', transform=transform)
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_loader = GeoDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = GeoDataLoader(val_dataset, batch_size=args.batch_size)
test_loader = GeoDataLoader(test_dataset, batch_size=args.batch_size)

# 定义 ConvNeXt 图像模型
class ConvNeXtImageModel(nn.Module):
    def __init__(self):
        super(ConvNeXtImageModel, self).__init__()
        self.model = convnext_small(pretrained=True)
        in_features = self.model.classifier[2].in_features
        self.model.classifier[2] = nn.Identity()  # 移除分类层

    def forward(self, x):
        return self.model(x)



class GCNGraphModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCNGraphModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        # 使用全局平均池化，获取每个图的嵌入
        graph_embedding = global_mean_pool(x, batch)
        return graph_embedding

class MultiModalContrastiveModel(nn.Module):
    def __init__(self, image_embed_dim, graph_embed_dim, hidden_dim):
        super(MultiModalContrastiveModel, self).__init__()
        self.image_model = ConvNeXtImageModel()
        self.graph_model = GCNGraphModel(input_dim=3, hidden_dim=64, output_dim=graph_embed_dim)
        
        # 添加降维层，将图像嵌入从 768 降到 128
        self.image_fc = nn.Linear(image_embed_dim, graph_embed_dim)

    def forward(self, image, graph_data):
        image_embedding = self.image_model(image)
        image_embedding = self.image_fc(image_embedding)  # 降维到 128
        graph_embedding = self.graph_model(graph_data)
        return image_embedding, graph_embedding

# 初始化模型和优化器
model = MultiModalContrastiveModel(image_embed_dim=768, graph_embed_dim=128, hidden_dim=256).to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)


def contrastive_loss(image_embeddings, graph_embeddings, labels, temperature=0.5):
    # 计算余弦相似度
    similarities = F.cosine_similarity(image_embeddings, graph_embeddings)
    
    # 对相似度应用温度缩放
    similarities = similarities / temperature
    
    # 使用二元交叉熵损失
    labels = labels.float()  # 将标签转换为浮点型
    loss = F.binary_cross_entropy_with_logits(similarities, labels)
    return loss

# 训练函数
def train():
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    for images, graph_data, labels in train_loader:
        images = images.to(device)
        graph_data = graph_data.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # 前向传播
        image_embeddings, graph_embeddings = model(images, graph_data)

        # 计算损失
        loss = contrastive_loss(image_embeddings, graph_embeddings, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        # 计算准确率
        similarities = F.cosine_similarity(image_embeddings, graph_embeddings)
        preds = (similarities > 0).long()  # 相似度大于 0 的为正类（1），否则为负类（0）
        
        #print(f"[DEBUG] Prediction: {preds}")
        #print(f"[DEBUG] Label: {labels}")
        
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    avg_loss = total_loss / len(train_loader.dataset)
    accuracy = total_correct / total_samples
    return avg_loss, accuracy

# 测试函数
@torch.no_grad()
def test(data_loader):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, graph_data, labels in data_loader:
            images = images.to(device)
            graph_data = graph_data.to(device)
            labels = labels.to(device)

            # 前向传播
            image_embeddings, graph_embeddings = model(images, graph_data)

            # 计算损失，传入 labels 参数
            loss = contrastive_loss(image_embeddings, graph_embeddings, labels)
            total_loss += loss.item() * images.size(0)

            # 计算准确率
            similarities = F.cosine_similarity(image_embeddings, graph_embeddings)
            preds = (similarities > 0).long()
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / len(data_loader.dataset)
    accuracy = total_correct / total_samples
    return avg_loss, accuracy

# 训练和验证流程
best_val_acc = 0.0
best_test_acc = 0.0
for epoch in range(args.epochs):
    train_loss, train_acc = train()
    val_loss, val_acc = test(val_loader)
    test_loss, test_acc = test(test_loader)

    # 更新最优验证准确率及对应的测试准确率
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_test_acc = test_acc
        print(f"=====Epoch {epoch}: New best validation acc=======")

    print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}")

# 最终测试
print(f"Best Validation Accuracy: {best_val_acc:.4f}, Corresponding Test Accuracy: {best_test_acc:.4f}")