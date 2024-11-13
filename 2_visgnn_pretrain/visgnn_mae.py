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
parser.add_argument('--image_size', type=int, default=224)
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

###################################################################################################################
def mask_image(images, mask_ratio=0.75, patch_size=16):
    # 获取批次大小、通道数、高度和宽度
    batch_size, c, h, w = images.shape
    #print("1:", images.shape) # (B C H W) 100 3 64 64 
    num_patches = (h // patch_size) * (w // patch_size)
    num_masked = int(mask_ratio * num_patches)

    # 将图像分割成 patch
    patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    #print("2:", patches.shape) # (B C nH nW P P) 100 3 4 4 16 16
    patches = patches.contiguous().view(batch_size, c, -1, patch_size, patch_size)
    #print("3:", patches.shape) # (B C nP P P) 100 3 16 16 16

    # 随机选择要掩盖的 patch 并设置为 0
    masked_patches = patches.clone()
    indices = torch.randperm(patches.size(2))[:num_masked]
    masked_patches[:, :, indices] = 0
    return masked_patches, indices

class MaskedAutoencoder(nn.Module):
    def __init__(self, patch_size=16, num_patches=196):
        super(MaskedAutoencoder, self).__init__()
        #print(f"Patch size: {patch_size}")
        self.num_patches = num_patches

        # 将ResNet18改为ResNet50
        resnet_base = models.resnet50(pretrained=True)
        resnet_base.fc = nn.Identity()
        self.encoder = resnet_base

        # ResNet50的输出维度是2048（而不是ResNet18的512）
        hidden_dim = 1024  # 添加中间隐藏层维度

        self.decoder = nn.Sequential(
            # 2048 -> 1024
            nn.Linear(2048, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            
            # 1024 -> 512
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            
            # 512 -> 256
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.ReLU(),
            
            # 256 -> patch_size * patch_size * 3
            nn.Linear(hidden_dim // 4, patch_size * patch_size * 3),
            nn.LayerNorm(patch_size * patch_size * 3),
            nn.ReLU(),
            
            # 最后的重建层
            nn.Linear(patch_size * patch_size * 3, patch_size * patch_size * 3)
        )

    def forward(self, patches, masked_indices, patch_siz=16):
        # (B C nP P P) 100, 3, 16, 16, 16
        # After view: (B * nP, C, P, P) = (B * nP, 3, 16, 16) = (1600 3 16 16)
        encoded_patches = self.encoder(patches.view(-1, patches.size(1), patch_siz, patch_siz))
        # After encoder: (B * nP, 2048) 1600, 2048 注意这里维度变成了2048
        #print("4:", encoded_patches.shape)

        encoded_patches = encoded_patches.view(-1, self.num_patches, encoded_patches.size(-1))
        # After view: (B, nP, 2048) 100, 16, 2048
        #print("5:", encoded_patches.shape)
        masked_patches = encoded_patches[:, masked_indices]
        reconstructed_patches = self.decoder(masked_patches)
        #print("9:", reconstructed_patches.shape) # 100, 12, 768
        return reconstructed_patches


def pretrain_mae():
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    mae_model = MaskedAutoencoder(patch_size=16).to(device)  # 确保这里传递了 patch_size
    optimizer = optim.Adam(mae_model.parameters(), lr=args.lr)

    for epoch in range(args.pretrain_epochs):
        mae_model.train()
        total_loss = 0
        for images, _ in train_loader:
            images = images.to(device)
            optimizer.zero_grad()
            masked_patches, masked_indices = mask_image(images)
            #print("6", masked_patches.shape) # (B, C, nP, P, P) 100, 3, 16, 16, 16

            masked_patches, masked_indices = masked_patches.to(device), masked_indices.to(device)
            reconstructed_patches = mae_model(masked_patches, masked_indices) 
            #print("10", masked_indices.shape)

            marked_ = masked_patches[:,:,masked_indices]
            #print("8", marked_.shape) # 100, 3, 12, 16, 16

            marked_ = marked_.view(100, 147, 3 * 16 * 16)
            
            loss = F.mse_loss(reconstructed_patches, marked_) 
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        #print(f"Pretrain Epoch {epoch+1}/{args.pretrain_epochs}, Loss: {total_loss/len(train_loader):.4f}")
    return mae_model  # 如果需要返回预训练模型
###################################################################################################################

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
    pretrained_base_model = pretrain_mae()
    if args.model_type == 'resnet':
        model = ResNetClassifier(num_classes=num_class).to(device)
        model.model.load_state_dict(pretrained_base_model.state_dict(), strict=False)
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

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
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