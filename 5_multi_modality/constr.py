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
parser.add_argument('--batch_size', type=int, default=64)
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

# 定义 GCN 图结构模型
class GCNGraphModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCNGraphModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x.mean(dim=0)  # 使用全局平均池化

# 多模态对比学习模型
class MultiModalContrastiveModel(nn.Module):
    def __init__(self, image_embed_dim, graph_embed_dim, hidden_dim):
        super(MultiModalContrastiveModel, self).__init__()
        self.image_model = ConvNeXtImageModel()
        self.graph_model = GCNGraphModel(input_dim=3, hidden_dim=64, output_dim=graph_embed_dim)
        self.fc = nn.Linear(image_embed_dim + graph_embed_dim, hidden_dim)

    def forward(self, image, graph_data):
        image_embedding = self.image_model(image)
        graph_embedding = self.graph_model(graph_data)
        combined_embedding = torch.cat((image_embedding, graph_embedding), dim=1)
        return self.fc(combined_embedding)

# 初始化模型和优化器
model = MultiModalContrastiveModel(image_embed_dim=768, graph_embed_dim=128, hidden_dim=256).to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)

# 对比学习损失函数（InfoNCE Loss）
def contrastive_loss(image_embeddings, graph_embeddings, temperature=0.5):
    similarities = F.cosine_similarity(image_embeddings.unsqueeze(1), graph_embeddings.unsqueeze(0), dim=2)
    labels = torch.arange(image_embeddings.size(0)).to(device)
    loss = F.cross_entropy(similarities / temperature, labels)
    return loss

# 训练函数
def train():
    model.train()
    total_loss = 0
    for images, graph_data, _ in train_loader:
        images = images.to(device)
        graph_data = graph_data.to(device)

        optimizer.zero_grad()
        image_embeddings = model.image_model(images)
        graph_embeddings = model.graph_model(graph_data)
        loss = contrastive_loss(image_embeddings, graph_embeddings)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(train_loader.dataset)

# 测试函数
@torch.no_grad()
def test(loader):
    model.eval()
    total_loss = 0
    for images, graph_data, _ in loader:
        images = images.to(device)
        graph_data = graph_data.to(device)

        image_embeddings = model.image_model(images)
        graph_embeddings = model.graph_model(graph_data)
        loss = contrastive_loss(image_embeddings, graph_embeddings)
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)

# 训练和验证流程
best_val_loss = float('inf')
for epoch in range(args.epochs):
    train_loss = train()
    torch.cuda.empty_cache()  
    val_loss = test(val_loader)
    torch.cuda.empty_cache()  

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        print(f"Epoch {epoch}: New best validation loss: {val_loss:.4f}")

    print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

# 最终测试
test_loss = test(test_loader)
torch.cuda.empty_cache()  
print(f"Test Loss: {test_loss:.4f}")