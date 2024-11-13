import argparse
import os
import os.path as osp
import time
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.models import resnet18, resnet50, vit_b_16, resnet101, vit_b_32, convnext_small
from torchvision import models, transforms

import random
import numpy as np
import csv

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)

def get_augmentations_str(args):
    augmentations = []
    if args.center_crop:
        augmentations.append("center_crop")
    if args.horizontal_flip:
        augmentations.append("horizontal_flip")
    if args.rotation:
        augmentations.append("rotation")
    if args.affine:
        augmentations.append("affine")
    if args.perspective:
        augmentations.append("perspective")
    if args.normalize:
        augmentations.append("normalize")
    
    return "_".join(augmentations) if augmentations else "no_augmentations"

# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='PROTEINS')
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--image_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=5e-6)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--seed', type=int, default=2)
parser.add_argument('--model_type', type=str, choices=['resnet', 'vit', 'convnext_base'], default='convnext_base')
parser.add_argument('--center_crop', action='store_true', help='Apply Center Crop')
parser.add_argument('--horizontal_flip', action='store_true', help='Apply Random Horizontal Flip')
parser.add_argument('--rotation', action='store_true', help='Apply Random Rotation')
parser.add_argument('--affine', action='store_true', help='Apply Random Affine Transformation')
parser.add_argument('--perspective', action='store_true', help='Apply Random Perspective Transformation')
parser.add_argument('--normalize', action='store_true', help='Apply Normalization')
parser.add_argument('--pretrain_epochs', type=int, default=10, help='Number of epochs for pretraining')

args = parser.parse_args()

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

set_seed(args.seed)

# 自定义数据集
class ImageDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
        image = Image.open(img_name).convert("RGB")
        label = int(self.annotations.iloc[idx, 1])

        if self.transform:
            image = self.transform(image)

        return image, label

# 数据变换
def get_transforms(args):
    transform_list = [transforms.Resize((args.image_size, args.image_size))]

    if args.center_crop:
        transform_list.append(transforms.CenterCrop(args.image_size))
    if args.horizontal_flip:
        transform_list.append(transforms.RandomHorizontalFlip())
    if args.rotation:
        transform_list.append(transforms.RandomRotation(90))
    if args.affine:
        transform_list.append(transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10))
    if args.perspective:
        transform_list.append(transforms.RandomPerspective(distortion_scale=0.5, p=0.5))

    transform_list.append(transforms.ToTensor())

    if args.normalize:
        transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

    return transforms.Compose(transform_list)

train_transform = get_transforms(args)
test_transform = transforms.Compose([
    transforms.Resize((args.image_size, args.image_size)),
    transforms.ToTensor(),
])

# 加载数据集
dataset = ImageDataset(csv_file=f'./img/{args.dataset}/dataset.csv', 
                        root_dir='', transform=train_transform)

# 数据集划分
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

def get_num_class(dataset_name):
    if dataset_name == 'IMDB-MULTI':
        return 3
    else:
        return 2

num_class = get_num_class(args.dataset)

train_indices, val_indices, test_indices = random_split(list(range(len(dataset))), [train_size, val_size, test_size])

train_dataset = torch.utils.data.Subset(dataset, train_indices)
val_dataset = torch.utils.data.Subset(dataset, val_indices)
test_dataset = torch.utils.data.Subset(dataset, test_indices)
pretrain_dataset = ImageDataset(
    csv_file='/mnt/data1/zhaoxinjian/zhongkai/TMLR_VISG_/pretrain/graphs/dataset.csv',
    root_dir='',
)

train_dataset.transform = train_transform
val_dataset.transform = test_transform
test_dataset.transform = test_transform

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

# 拼图预训练任务
class JigsawDataset(Dataset):
    def __init__(self, original_dataset, num_patches=9):
        self.dataset = original_dataset
        self.num_patches = num_patches

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        patches = self.create_jigsaw_puzzle(img)
        order = torch.randperm(self.num_patches)
        shuffled_patches = patches[order]
        return shuffled_patches, order

    def create_jigsaw_puzzle(self, img):
        _, h, w = img.shape
        patch_size = h // int(self.num_patches**0.5)
        patches = img.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
        patches = patches.contiguous().view(3, -1, patch_size, patch_size)
        patches = patches.permute(1, 0, 2, 3)
        return patches

class JigsawModel(nn.Module):
    def __init__(self, base_model, num_patches=9):
        super(JigsawModel, self).__init__()
        self.base_model = base_model
        self.fc = nn.Linear(1000, num_patches)#####

    def forward(self, x):
        batch_size, num_patches, c, h, w = x.shape
        x = x.view(batch_size * num_patches, c, h, w)
        features = self.base_model(x)  # 检查这里的输出维度
        #print("Features shape after base_model:", features.shape)  # 添加调试信息
        features = features.view(batch_size, num_patches, -1)
        #print("Features shape after view:", features.shape)  # 添加调试信息
        output = self.fc(features)
        return output

# 预训练函数
def pretrain():
    jigsaw_dataset = JigsawDataset(pretrain_dataset)
    jigsaw_loader = DataLoader(jigsaw_dataset, batch_size=args.batch_size, shuffle=True)

    base_model = models.resnet50(pretrained=True)
    jigsaw_model = JigsawModel(base_model).to(device)
    optimizer = optim.Adam(jigsaw_model.parameters(), lr=args.lr)

    for epoch in range(args.pretrain_epochs):
        jigsaw_model.train()
        total_loss = 0
        for patches, order in jigsaw_loader:
            patches, order = patches.to(device), order.to(device)
            optimizer.zero_grad()
            outputs = jigsaw_model(patches)
            loss = F.cross_entropy(outputs.view(-1, 9), order.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Pretrain Epoch {epoch+1}/{args.pretrain_epochs}, Loss: {total_loss/len(jigsaw_loader):.4f}")

    return base_model

# 模型定义
class ViTClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super(ViTClassifier, self).__init__()
        self.model = vit_b_32(pretrained=True)
        in_features = self.model.heads.head.in_features
        self.model.heads.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        return self.model(x)

class ResNetClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super(ResNetClassifier, self).__init__()
        self.model = resnet50(pretrained=True)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(in_features, num_classes),
        )
    def forward(self, x):
        return self.model(x)

# 训练函数
def train(model, optimizer):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(train_loader.dataset)

# 测试函数
@torch.no_grad()
def test(model, loader):
    model.eval()
    total_correct = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        total_correct += (preds == labels).sum().item()
    return total_correct / len(loader.dataset)

# 主训练过程
def main():
    # 预训练
    pretrained_base_model = pretrain()

    # 初始化分类器
    if args.model_type == 'resnet':
        model = ResNetClassifier(num_classes=num_class).to(device)
        model.model.load_state_dict(pretrained_base_model.state_dict(), strict=False)
    elif args.model_type == 'vit':
        model = ViTClassifier(num_classes=num_class).to(device)
        # 对于ViT，你可能需要调整预训练模型的加载方式
    elif args.model_type == 'convnext_base':
        model = convnext_small(pretrained=True)
        model.classifier[2] = nn.Sequential(
            nn.Linear(model.classifier[2].in_features, model.classifier[2].in_features//2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(model.classifier[2].in_features//2, num_class),
        )
        model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val_acc = 0.0
    best_test_acc = 0.0
    times = []
    patience = 15
    no_improve_epochs = 0

    augmentations_str = get_augmentations_str(args)
    filename = f'./results/{args.model_type}_dataset={args.dataset}_batch_size={args.batch_size}_lr={args.lr}_epochs={args.epochs}_{augmentations_str}.csv'
    
    with open(filename, 'a', newline='') as csvfile:
        fieldnames = ['Model Type', 'Seed', 'Epoch', 'Loss', 'Train Acc', 'Val Acc', 'Test Acc']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if os.stat(filename).st_size == 0:
            writer.writeheader()
        
        for epoch in range(1, args.epochs + 1):
            start = time.time()
            loss = train(model, optimizer)
            train_acc = test(model, train_loader)
            val_acc = test(model, val_loader)
            test_acc = test(model, test_loader)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
            
            print(f"Epoch {epoch}/{args.epochs}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}")
            times.append(time.time() - start)
            
            writer.writerow({'Model Type': args.model_type, 'Seed': args.seed, 'Epoch': epoch, 'Loss': loss, 'Train Acc': train_acc, 'Val Acc': val_acc, 'Test Acc': test_acc})
            
            if no_improve_epochs >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f'Best Validation Acc: {best_val_acc:.4f}, Corresponding Test Acc: {best_test_acc:.4f}')
    print(f'Median time per epoch: {torch.tensor(times).median():.4f}s')

if __name__ == "__main__":
    main()