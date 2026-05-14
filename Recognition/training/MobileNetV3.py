import os
import sys
import argparse
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score

torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = r'/root/Recognition/dataset/train'
OUT_PUT_MODEL_FT = '/root/Recognition/models/mobilenet.pth'  # 微调后的模型保存路径
EPOCHS = 150
BATCH_SIZE = 32
LEARNING_RATE = 0.0001

def get_data_transforms(data_dir):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet的均值和标准差
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet的均值和标准差
    ])
    dataset = datasets.ImageFolder(data_dir, transform=train_transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    val_dataset.dataset.transform = val_transform
    return train_dataset, val_dataset

def setup_to_finetune(model):
    # 冻结所有层
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True
    # 修改全连接层
    model.classifier[3] = nn.Linear(in_features=1280, out_features=7)
    return model

def plot_training(history):
    acc = history['train_acc']
    val_acc = history['val_acc']
    loss = history['train_loss']
    val_loss = history['val_loss']
    f1 = history['train_f1']
    val_f1 = history['val_f1']
    epochs = range(len(acc))

    plt.figure(figsize=(10, 8))
    plt.plot(epochs, acc, 'b-', label='Train Accuracy')
    plt.plot(epochs, val_acc, 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    plt.savefig("/root/Recognition/figs/MobileNetV3/accuracy.png")

    plt.figure(figsize=(10, 8))
    plt.plot(epochs, loss, 'b-', label='Train Loss')
    plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.savefig("/root/Recognition/figs/MobileNetV3/loss.png")

    plt.figure(figsize=(10, 8))
    plt.plot(epochs, f1, 'b-', label='Train F1_score')
    plt.plot(epochs, val_f1, 'r-', label='Validation F1_score')
    plt.title('Training and Validation F1_score')
    plt.xlabel('Epochs')
    plt.ylabel('F1_score')
    plt.legend()
    plt.grid()
    plt.savefig("/root/Recognition/figs/MobileNetV3/f1_score.png")

# 训练函数
def train(args):
    epochs = int(args.epoch)
    batch_size = int(args.batch_size)

    train_dataset, val_dataset = get_data_transforms(args.data_dir)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model = models.mobilenet_v3_large(pretrained=True)
    model = setup_to_finetune(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=10, verbose=True)
    model.to(device)

    # 训练模型
    best_val_acc = 0.0
    best_model = None
    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = 20  # 定义早停的耐心值

    print("Start Training")
    history = {'train_acc': [], 'val_acc': [], 'train_loss': [], 'val_loss': [], 'train_f1': [], 'val_f1': []}

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        all_train_preds = []
        all_train_labels = []

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            all_train_preds.extend(predicted.cpu().numpy())
            all_train_labels.extend(labels.cpu().numpy())

        train_acc = correct_train / total_train
        train_loss = running_loss / len(train_loader)
        train_f1 = f1_score(all_train_labels, all_train_preds, average='macro')  # 宏平均 F1 分数
        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['train_f1'].append(train_f1)

        model.eval()
        correct_val = 0
        total_val = 0
        val_loss = 0.0
        all_val_preds = []
        all_val_labels = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                all_val_preds.extend(predicted.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())

        val_acc = correct_val / total_val
        val_loss = val_loss / len(val_loader)
        val_f1 = f1_score(all_val_labels, all_val_preds, average='macro')  # 宏平均 F1 分数
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)
        history['val_f1'].append(val_f1)

        # 输出训练和验证的损失与准确率
        print(f"[Epoch {epoch + 1}/{epochs}] | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_val_acc = val_acc
            best_model = model
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

        # 学习率调整
        scheduler.step(val_loss)

    print("Finished Training")
    print(f"Best Val Acc: {best_val_acc:.4f}. Saving best model...")
    torch.save(best_model, args.output_model_file)
    print(f"Best model saved to {args.output_model_file}")

    plot_training(history)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=DATA_DIR, help="Data directory path")
    parser.add_argument("--epoch", default=EPOCHS, help="Number of epochs")
    parser.add_argument("--batch_size", default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--output_model_file", default=OUT_PUT_MODEL_FT, help="Output model file")
    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        print("Data directory not found")
        sys.exit(1)

    train(args)