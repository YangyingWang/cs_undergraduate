import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              confusion_matrix, classification_report, precision_recall_curve, average_precision_score)
import argparse
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TEST_DIR = r'D:/Major/AIProject/Recognition/dataset/test'
MODEL_DIR = 'D:/Major/AIProject/Recognition/models/resnet50.pth'

def load_model(model_path):
    model = torch.load(model_path, map_location = device)
    model.to(device)
    model.eval()
    return model

def get_test_data(test_dir):
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet的均值和标准差
    ])
    
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)
    
    return test_loader, test_dataset.classes

def plot_test(results, class_names):
    all_logits = results['all_logits']
    all_labels = results['all_labels']
    all_probs = torch.softmax(torch.tensor(all_logits), dim=1).numpy()  # 将 logits 转换为概率

    plt.figure(figsize=(12, 8))
    for i, class_name in enumerate(class_names):
        true_labels = (all_labels == i).astype(int)
        pred_probs = all_probs[:, i]

        precision, recall, _ = precision_recall_curve(true_labels, pred_probs)
        ap = average_precision_score(true_labels, pred_probs)
        plt.plot(recall, precision, label=f'{class_name} (AP = {ap:.2f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'PR Curve for Classes')
    plt.legend(loc='best')
    plt.grid()
    plt.savefig("D:/Major/AIProject/Recognition/figs/ResNet50/pr_curve.png")

    plt.figure(figsize=(10, 8))
    sns.heatmap(results['cm'], annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.savefig("D:/Major/AIProject/Recognition/figs/ResNet50/confusion_matrix.png")

    report = classification_report(all_labels, results['all_preds'], target_names=class_names)
    output_path = "D:/Major/AIProject/Recognition/figs/ResNet50/report.txt"
    with open(output_path, "w") as f:
        f.write(report)
        f.write("\n\n")
        f.write(f"Accuracy: {results['acc']:.4f}\n")
        f.write(f"Precision: {results['pre']:.4f}\n")
        f.write(f"Recall: {results['recall']:.4f}\n")
        f.write(f"F1 Score: {results['f1']:.4f}\n")

def test(args):
    model = load_model(args.model_file)
    test_loader, class_names = get_test_data(args.test_dir)
    print("Start Test")
    all_preds = []
    all_labels = []
    all_logits = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_logits.extend(outputs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_logits = np.array(all_logits)

    acc = accuracy_score(all_labels, all_preds)
    pre = precision_score(all_labels, all_preds, average='weighted')  # 根据需要调整平均方式
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)

    results = { 'acc' : acc,
              'pre': pre,
              'recall': recall,
              'f1': f1,
              'cm' : cm,
              'all_labels': all_labels,
              'all_preds': all_preds,
              'all_logits': all_logits
            }
    print("Finished Test")
    print("Start plot")
    plot_test(results, class_names)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_file", default=MODEL_DIR, help="Path to the saved model file")
    parser.add_argument("--test_dir", default=TEST_DIR, help="Path to the test dataset directory")
    args = parser.parse_args()

    if not os.path.exists(args.model_file):
        print(f"Error: Model file '{args.model_file}' not found.")
        sys.exit(1)
    if not os.path.exists(args.test_dir):
        print(f"Error: Test directory '{args.test_dir}' not found.")
        sys.exit(1)

    test(args)
