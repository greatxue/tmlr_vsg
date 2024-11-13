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
from torchvision.models import resnet18, resnet50, vit_b_16, resnet101,vit_b_32,convnext_small
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
parser.add_argument('--model_type', type=str, choices=['resnet', 'vit','convnext_small'], default='convnext_base')
parser.add_argument('--center_crop', action='store_true', help='Apply Center Crop')
parser.add_argument('--horizontal_flip', action='store_true', help='Apply Random Horizontal Flip')
parser.add_argument('--rotation', action='store_true', help='Apply Random Rotation')
parser.add_argument('--affine', action='store_true', help='Apply Random Affine Transformation')
parser.add_argument('--perspective', action='store_true', help='Apply Random Perspective Transformation')
parser.add_argument('--normalize', action='store_true', help='Apply Normalization')


args = parser.parse_args()

# 设置设备
if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('cpu')
else:
    device = torch.device('cpu')

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

    transform_list.append(transforms.ToTensor())  # 将 ToTensor 放在所有图像增强操作之后

    if args.normalize:
        transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

    return transforms.Compose(transform_list)
train_transform = get_transforms(args)
test_transform = transforms.Compose([
    transforms.Resize((args.image_size, args.image_size)),
    transforms.ToTensor(),

    # transforms.CenterCrop(224),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 加载数据集
if args.dataset=='Cycles':
    dataset = ImageDataset(csv_file=f'./img/{args.dataset}/3/dataset.csv', 
                                root_dir='')
else:
    dataset = ImageDataset(csv_file=f'./img/{args.dataset}/dataset.csv', 
                            root_dir='',transform=train_transform)
    test_dataset = ImageDataset(csv_file=f'./img/{args.dataset}/dataset.csv', 
                            root_dir='',transform=train_transform)
    val_dataset = ImageDataset(csv_file=f'./img/{args.dataset}/dataset.csv', 
                            root_dir='',transform=train_transform)                           

# 数据集划分为训练集、验证集和测试集
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
if args.dataset=='MUTAG':
    test_size = int(0.1 * len(dataset))
    val_size = len(dataset) - train_size - val_size

def get_num_class(dataset_name):
    if dataset_name == 'IMDB-MULTI':
        return 3
    else:
        return 2

# 使用函数来获取num_class
num_class = get_num_class(args.dataset)

# train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# # 应用数据增强和预处理
# train_dataset.transform = train_transform
# val_dataset.transform = test_transform
# test_dataset.transform = test_transform
train_indices, val_indices, test_indices = random_split(list(range(len(dataset))), [train_size, val_size, test_size])

# 使用索引来创建新的Dataset实例（如果需要）
train_dataset = torch.utils.data.Subset(dataset, train_indices)
val_dataset = torch.utils.data.Subset(val_dataset, val_indices)
test_dataset = torch.utils.data.Subset(test_dataset, test_indices)

# 设置变换
train_dataset.transform = train_transform
val_dataset.transform = test_transform
test_dataset.transform = test_transform

# 创建DataLoader
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
class ViTClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super(ViTClassifier, self).__init__()
        self.model = vit_b_32(pretrained=True)
        in_features = self.model.heads.head.in_features
        self.model.heads.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, num_classes),
            # nn.ReLU(),
            # nn.Dropout(0.5),
            # nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.model(x)
# 定义模型
class ResNetClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super(ResNetClassifier, self).__init__()
        self.model = resnet50(pretrained=True)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            # nn.Dropout(0.5),  # 加入Dropout层
        nn.Linear(in_features, num_class),
        # nn.ReLU(),
        # nn.Dropout(0.5),
        # nn.Linear(in_features//2, num_class),   
        )
    def forward(self, x):
        return self.model(x)

# 初始化模型和优化器
# model = ResNetClassifier(num_classes=3).to(device)

if args.model_type == 'resnet':
    model = ResNetClassifier(num_classes=2).to(device)
    optimizer = optim.Adam([
        {'params': model.model.fc.parameters(), 'lr': 10*args.lr},  # 新添加层使用较大学习率
        {'params': [param for name, param in model.named_parameters() if "fc" not in name], 'lr': args.lr}  # 预训练层使用较小学习率
    ])
elif args.model_type == 'vit':
    model = ViTClassifier(num_classes=2).to(device)
    optimizer = optim.Adam([
        {'params': model.model.heads.head.parameters(), 'lr': 10*args.lr},  # 新添加层使用较大学习率
        {'params': [param for name, param in model.named_parameters() if "head" not in name], 'lr': args.lr}  # 预训练层使用较小学习率
    ])
elif args.model_type == 'convnext_base':
        model = convnext_small(pretrained=True)  # 加载预训练的 ConvNeXt-Small 模型

        # 获取模型最后一层的输入特征数
        in_features = model.classifier[2].in_features

        # 替换模型的最后一层，全连接层 (classifier)
        model.classifier[2] = nn.Sequential(
        # nn.Dropout(0.5),
        nn.Linear(in_features, in_features//2),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(in_features//2, num_class),
        )
        model=model.to(device)
optimizer = optim.Adam(model.parameters(), lr=10*args.lr, weight_decay=1e-4)

# 训练函数
def train():
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        # labels = labels.float().unsqueeze(-1)
        loss = F.cross_entropy(outputs, labels)#F.binary_cross_entropy(torch.sigmoid(outputs), labels)#
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(train_loader.dataset)

# 测试函数
@torch.no_grad()
def test(loader):
    model.eval()
    total_correct = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        total_correct += (preds == labels).sum().item()
    return total_correct / len(loader.dataset)

# 训练过程
best_val_acc = 0.0
best_test_acc = 0.0
times = []
patience = 15
no_improve_epochs = 0
set_seed(args.seed)

augmentations_str = get_augmentations_str(args)
filename = f'./results/{args.model_type}_dataset={args.dataset}_batch_size={args.batch_size}_lr={args.lr}_epochs={args.epochs}_{augmentations_str}.csv'
print(train_transform)
print(filename)

# Open the CSV file in append mode
with open(filename, 'a', newline='') as csvfile:
    fieldnames = ['Model Type', 'Seed', 'Epoch', 'Loss', 'Train Acc', 'Val Acc', 'Test Acc']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    # Write the header only if the file is empty
    if os.stat(filename).st_size == 0:
        writer.writeheader()
    
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        loss = train()
        train_acc = test(train_loader)
        val_acc = test(val_loader)
        test_acc = test(test_loader)
        
        # 更新最优验证准确率及对应的测试准确率
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
        
        print(f"Epoch {epoch}/{args.epochs}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}")
        times.append(time.time() - start)
        
        # Write the results to the CSV file

        # 检查是否需要提前停止
        if no_improve_epochs >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
    writer.writerow({'Model Type': args.model_type, 'Seed': args.seed, 'Epoch': epoch, 'Loss': loss, 'Val Acc': best_val_acc, 'Test Acc': best_test_acc})

print(f'Best Validation Acc: {best_val_acc:.4f}, Corresponding Test Acc: {best_test_acc:.4f}')
print(f'Median time per epoch: {torch.tensor(times).median():.4f}s')



# import argparse
# import os
# import os.path as osp
# import time
# import pandas as pd
# from PIL import Image
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader, random_split
# from torchvision import transforms
# from torchvision.models import resnet18,resnet50,vit_b_16,resnet101

# # 解析命令行参数
# parser = argparse.ArgumentParser()
# parser.add_argument('--dataset', type=str, default='IMDB-BINARY')
# parser.add_argument('--batch_size', type=int, default=32)
# parser.add_argument('--lr', type=float, default=0.0001)
# parser.add_argument('--epochs', type=int, default=100)
# args = parser.parse_args()

# # 设置设备
# if torch.cuda.is_available():
#     device = torch.device('cuda')
# elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
#     device = torch.device('cpu')
# else:
#     device = torch.device('cpu')

# # 自定义数据集
# class MUTAGImageDataset(Dataset):
#     def __init__(self, csv_file, root_dir, transform=None):
#         self.annotations = pd.read_csv(csv_file)
#         self.root_dir = root_dir
#         self.transform = transform

#     def __len__(self):
#         return len(self.annotations)

#     def __getitem__(self, idx):
#         img_name = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
#         image = Image.open(img_name).convert("RGB")
#         label = int(self.annotations.iloc[idx, 1])

#         if self.transform:
#             image = self.transform(image)

#         return image, label

# # 数据变换
# train_transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.RandomResizedCrop(224),
#     transforms.RandomHorizontalFlip(),  # 随机水平翻转
#     transforms.RandomRotation(180), 
#     transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),  # 平移、缩放、剪切
#     transforms.RandomPerspective(distortion_scale=0.5, p=0.5),  # 随机透视变换
#     transforms.ToTensor(),
#     # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

# test_transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     # transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])


# # transform = transforms.Compose([
# #     transforms.Resize((224, 224)),
# #     transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
# #     transforms.RandomHorizontalFlip(),
# #     transforms.ToTensor(),
# #     # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# # ])

# # 加载数据集
# dataset = MUTAGImageDataset(csv_file=f'./img/{args.dataset}/dataset.csv', 
#                             root_dir='', 
#                             )

# # 数据集划分
# train_size = int(0.9 * len(dataset))
# test_size = len(dataset) - train_size
# train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# train_dataset.dataset.transform = train_transform
# test_dataset.dataset.transform = test_transform

# train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

# # 定义模型
# class ResNetClassifier(nn.Module):
#     def __init__(self, num_classes=3):
#         super(ResNetClassifier, self).__init__()
#         self.model = resnet101(pretrained=True)
#         in_features = self.model.fc.in_features
#         self.model.fc = nn.Sequential(
#             nn.Dropout(0.5),  # 加入Dropout层
#             nn.Linear(in_features, num_classes)
#         )
#     def forward(self, x):
#         return self.model(x)

# # 初始化模型和优化器
# model = ResNetClassifier(num_classes=2).to(device)
# # class ViTModel(nn.Module):
# #     def __init__(self, num_classes=3):
# #         super(ViTModel, self).__init__()
# #         # 加载ViT模型
# #         self.model = vit_b_16(pretrained=True)
        
# #         # 获取原始分类头的输入特征数
# #         in_features = self.model.heads.head.in_features
        
# #         # 替换分类头，加入Dropout层和自定义全连接层
# #         self.model.heads.head = nn.Sequential(
# #             nn.Dropout(0.2),
# #             nn.Linear(in_features, num_classes)
# #         )
    
# #     def forward(self, x):
# #         return self.model(x)

# # 初始化模型
# # model = ViTModel(num_classes=3).to(device)

# optimizer = optim.Adam(model.parameters(), lr=args.lr,weight_decay=1e-4)
# # class CustomCNN(nn.Module):
# #     def __init__(self, num_classes=2):
# #         super(CustomCNN, self).__init__()
# #         self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
# #         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
# #         self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
# #         self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
# #         self.fc1 = nn.Linear(128 * 28 * 28, 512)
# #         self.fc2 = nn.Linear(512, num_classes)
# #         self.dropout = nn.Dropout(0.5)

# #     def forward(self, x):
# #         x = self.pool(F.relu(self.conv1(x)))
# #         x = self.pool(F.relu(self.conv2(x)))
# #         x = self.pool(F.relu(self.conv3(x)))
# #         x = x.view(-1, 128 * 28 * 28)
# #         x = F.relu(self.fc1(x))
# #         x = self.dropout(x)
# #         x = self.fc2(x)
# #         return x

# # model = CustomCNN(num_classes=3).to(device)
# # optimizer = optim.Adam(model.parameters(), lr=args.lr)
# # 训练函数
# def train():
#     model.train()
#     total_loss = 0
#     for images, labels in train_loader:
#         images, labels = images.to(device), labels.to(device)
#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = F.cross_entropy(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item() * images.size(0)
#     return total_loss / len(train_loader.dataset)

# # 测试函数
# @torch.no_grad()
# def test(loader):
#     model.eval()
#     total_correct = 0
#     for images, labels in loader:
#         images, labels = images.to(device), labels.to(device)
#         outputs = model(images)
#         _, preds = torch.max(outputs, 1)
#         # print(preds)
#         total_correct += (preds == labels).sum().item()
#     return total_correct / len(loader.dataset)

# # 训练过程
# times = []
# for epoch in range(1, args.epochs + 1):
#     start = time.time()
#     loss = train()
#     train_acc = test(train_loader)
#     test_acc = test(test_loader)
#     print(f"Epoch {epoch}/{args.epochs}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
#     times.append(time.time() - start)

# print(f'Median time per epoch: {torch.tensor(times).median():.4f}s')