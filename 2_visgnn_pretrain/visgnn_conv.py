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
from torchvision import transforms, models
from torchvision.models import resnet50, vit_b_32, convnext_small
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
    #if args.center_crop:
    #    augmentations.append("center_crop")
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
parser.add_argument('--image_size', type=int, default=224)
parser.add_argument('--lr', type=float, default=1e-6)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--seed', type=int, default=2)
parser.add_argument('--model_type', type=str, choices=['resnet', 'vit', 'convnext_base'], default='convnext_base')
parser.add_argument('--horizontal_flip', action='store_true', help='Apply Random Horizontal Flip')
parser.add_argument('--rotation', action='store_true', help='Apply Random Rotation')
parser.add_argument('--affine', action='store_true', help='Apply Random Affine Transformation')
parser.add_argument('--perspective', action='store_true', help='Apply Random Perspective Transformation')
parser.add_argument('--normalize', action='store_true', help='Apply Normalization')
parser.add_argument('--pretrain_epochs', type=int, default=5, help='Number of epochs for pretraining')

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
set_seed(args.seed)

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

def get_transforms(args):
    transform_list = [transforms.Resize((args.image_size, args.image_size))]
    #if args.center_crop:
    #    transform_list.append(transforms.CenterCrop(args.image_size))
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

dataset = ImageDataset(csv_file=f'./img/{args.dataset}/dataset.csv', root_dir='', transform=train_transform)
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

def get_num_class(dataset_name):
    return 3 if dataset_name == 'IMDB-MULTI' else 2

num_class = get_num_class(args.dataset)

train_indices, val_indices, test_indices = random_split(list(range(len(dataset))), [train_size, val_size, test_size])
train_dataset = torch.utils.data.Subset(dataset, train_indices)
val_dataset = torch.utils.data.Subset(dataset, val_indices)
test_dataset = torch.utils.data.Subset(dataset, test_indices)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

class ViTClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super(ViTClassifier, self).__init__()
        self.model = vit_b_32(pretrained=True)
        in_features = self.model.heads.head.in_features
        self.model.heads.head = nn.Sequential(nn.Dropout(0.5), nn.Linear(in_features, num_classes))

    def forward(self, x):
        return self.model(x)

class ResNetClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super(ResNetClassifier, self).__init__()
        self.model = resnet50(pretrained=True)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(nn.Linear(in_features, num_classes))

    def forward(self, x):
        return self.model(x)

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

def main():
    if args.model_type == 'resnet':
        model = ResNetClassifier(num_classes=num_class).to(device)
        #model.model.load_state_dict(pretrained_base_model.state_dict(), strict=False)
    elif args.model_type == 'vit':
        model = ViTClassifier(num_classes=num_class).to(device)
    elif args.model_type == 'convnext_base':
        model = convnext_small(pretrained=True)
        model.classifier[2] = nn.Sequential(
            nn.Linear(model.classifier[2].in_features, model.classifier[2].in_features//2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(model.classifier[2].in_features//2, num_class),
        )
        model = model.to(device)
#######################################################################################################################################
    pretrained_weights_path = '/mnt/data1/zhaoxinjian/zhongkai/TMLR_VISG_/convnext-v2-pytorch/weights/best_model.pth'  # 替换为您的预训练权重路径
    if os.path.exists(pretrained_weights_path):
        weights_dict = torch.load(pretrained_weights_path, map_location=device)
        model.load_state_dict(weights_dict, strict=False)  # 加载权重，允许不严格匹配
    else:
        print(f"Pretrained weights file not found at {pretrained_weights_path}")
########################################################################################################################################
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-3)#####################
    best_val_acc = 0.0
    best_test_acc = 0.0
    times = []
    patience = 15
    no_improve_epochs = 0

    augmentations_str = get_augmentations_str(args)
    filename = f'./results/{args.model_type}_dataset={args.dataset}_batch_size={args.batch_size}_lr={args.lr}_epochs={args.epochs}_{augmentations_str}.csv'
    
    with open(filename, 'a', newline='') as csvfile:
        fieldnames = ['Model Type', 'Seed', 'Best Epoch', 'Best Loss', 'Best Train Acc', 'Best Val Acc', 'Best Test Acc']
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
                best_results = {
                    'Model Type': args.model_type,
                    'Seed': args.seed,
                    'Best Epoch': epoch,
                    'Best Loss': loss,
                    'Best Train Acc': train_acc,
                    'Best Val Acc': val_acc,
                    'Best Test Acc': test_acc
                }
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
            
            print(f"Epoch {epoch}/{args.epochs}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}")
            times.append(time.time() - start)
            
            if no_improve_epochs >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        # 只写入最好的结果
        writer.writerow(best_results)

    print(f'Best Validation Acc: {best_val_acc:.4f}, Corresponding Test Acc: {best_test_acc:.4f}')
    print(f'Median time per epoch: {torch.tensor(times).median():.4f}s')

if __name__ == "__main__":
    main()