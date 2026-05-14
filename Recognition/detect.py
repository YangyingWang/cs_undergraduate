import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path):
    model = torch.load(model_path, map_location = device)
    model.to(device)
    model.eval()
    return model

def preprocess_image(image_path):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet的均值和标准差
    ])
    image = Image.open(image_path)
    image = preprocess(image)
    # 增加批次维度（模型的输入需要为 [Batch, Channel, Height, Width]）
    image = image.unsqueeze(0)
    return image

def predict(image_path, model, class_names):
    image = preprocess_image(image_path)
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)  # 转化为概率分布

    print("Class Probabilities:")
    for i, prob in enumerate(probabilities):
        print(f"Class: {class_names[i]}, Probability: {prob:.4f}")

    # 获取最高概率的类别索引和对应概率
    top_prob, top_class = torch.max(probabilities, dim=0)
    print(f"Predicted Class: {class_names[top_class]} with Probability: {top_prob.item():.4f}")
    
    plt.figure(figsize=(8, 8))
    plt.pie(
        probabilities,
        labels=class_names,
        autopct='%1.1f%%',  # 显示百分比
        startangle=140
    )
    plt.title("Class Probabilities")
    plt.show()

    return class_names[top_class], top_prob.item()

if __name__ == "__main__":
    model_path = "D:/Major/AIProject/Recognition/models/resnet50(1).pth" 
    image_path = "D:/Major/AIProject/Recognition/dataset/test/0/img0_2400.jpg" 
    class_names = ["surprised", "fear", "disgust", "happy", "sad", "anger", "normal"]

    model = load_model(model_path)
    predicted_class, probability = predict(image_path, model, class_names)
