import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import os

# 超参数设置
batch_size = 32
learning_rate = 1e-3
temperature = 0.5
num_epochs = 100

# 图像变换
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 自定义数据集类
class GraphImageDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 0]
        label = self.data.iloc[idx, 1]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# 加载数据集
dataset = GraphImageDataset(csv_file="./img/PROTEINS_multi/dataset.csv", transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 定义图像编码器（ResNet18 作为 backbone）
class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        self.encoder = resnet18(pretrained=True)
        self.encoder.fc = nn.Identity()  # 去掉最后的全连接层

    def forward(self, x):
        return self.encoder(x)

# SimCLR 对比学习模型
class SimCLR(nn.Module):
    def __init__(self, base_encoder, projection_dim=128):
        super(SimCLR, self).__init__()
        self.encoder = base_encoder
        # 投影头
        self.projector = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim)
        )

    def forward(self, x):
        features = self.encoder(x)
        projections = self.projector(features)
        return nn.functional.normalize(projections, dim=1)

# 对比损失（NT-Xent Loss）
def contrastive_loss(embeddings, temperature):
    # 计算相似度矩阵
    similarity_matrix = torch.matmul(embeddings, embeddings.T)
    labels = torch.arange(embeddings.size(0)).to(embeddings.device)
    mask = torch.eye(labels.size(0), dtype=torch.bool).to(embeddings.device)

    # 计算相似度
    positives = similarity_matrix[mask].view(labels.size(0), -1)
    negatives = similarity_matrix[~mask].view(labels.size(0), -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(labels.size(0), dtype=torch.long).to(embeddings.device)

    # 计算 NT-Xent Loss
    logits = logits / temperature
    loss = nn.CrossEntropyLoss()(logits, labels)
    return loss

# 初始化模型和优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimCLR(ImageEncoder()).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练过程
for epoch in range(num_epochs):
    total_loss = 0
    model.train()
    
    for images, _ in dataloader:
        # 获取同一图的不同视角图像对
        images = images.to(device)
        
        # 前向传播
        embeddings = model(images)
        
        # 计算对比损失
        loss = contrastive_loss(embeddings, temperature)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

print("预训练完成！")