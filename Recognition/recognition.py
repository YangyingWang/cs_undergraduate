import io
import gradio as gr
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ["surprised", "fear", "disgust", "happy", "sad", "anger", "normal"]
model_paths = {
        "ResNet50": "D:/Major/AIProject/Recognition/models/resnet50(1).pth",
        "VGG16": "D:/Major/AIProject/Recognition/models/VGG16.pth",
        "EfficientNet-b3": "D:/Major/AIProject/Recognition/models/efficientnet.pth",
        "MobileNetV3": "D:/Major/AIProject/Recognition/models/mobilenet.pth"
    }

def create_table(probabilities, class_names):
    data = [(i, class_names[i], probabilities[i]) for i in range(len(probabilities))]
    return data

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

def predict(image_path, model):
    image = preprocess_image(image_path)
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)  # 转化为概率分布

    # 获取最高概率的类别索引和对应概率
    top_prob, top_class = torch.max(probabilities, dim=0)
    fig = plt.figure(figsize=(5, 5))
    plt.pie(probabilities, labels=class_names, autopct='%1.1f%%', startangle=140)
    plt.title("Class Probabilities")
    img_buffer = io.BytesIO()  # 将图像保存到内存字节流中
    fig.savefig(img_buffer, format='png') 
    img_buffer.seek(0)  # 重置指针位置
    plt.close(fig) # 关闭图像，释放内存

    return class_names[top_class], top_prob.item(), probabilities, img_buffer

# Gradio 应用函数
def custom_predict(img_path, selected_model):
    model_path = model_paths[selected_model]
    print(f"Uploaded image is saved at: {img_path}")
    
    model = load_model(model_path)
    predicted_class, probability, probabilities, fig = predict(img_path, model)
    answer = create_table(probabilities, class_names)
    score = f"Predicted Class: {predicted_class} with Probability: {probability:.4f}"
    with open('pie_chart.png', 'wb') as f:
        f.write(fig.getvalue())
    fig = Image.open(fig)
    return answer, score, fig

def clear_input():
    return None, [], "", None  # 对gr.Image返回None

# 使用 Gradio 创建 UI
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            model_selector = gr.Dropdown(
                choices=["ResNet50", "VGG16", "EfficientNet-b3", "MobileNetV3"],
                value="ResNet50",
                label="选择模型"
            )
            imagebox = gr.Image(type="filepath", label="选择图片")

    with gr.Row():
        clear = gr.Button("clear")
        submit = gr.Button("submit")

    with gr.Row():
        with gr.Column():
            score = gr.Label(label="最高概率的类别索引和对应概率")

    with gr.Row():
        with gr.Column(scale=1):
            answer = gr.Dataframe(headers=["Index", "Class", "Probability"],label="预测类别概率")

    with gr.Row():      
        with gr.Column():
            output_img = gr.Image(type="filepath", label="预测类别概率分布")

    submit.click(fn=custom_predict, inputs=[imagebox, model_selector], outputs=[answer, score, output_img])
    clear.click(fn=clear_input, inputs=[], outputs=[imagebox, answer, score, output_img])

demo.launch(server_name="127.0.0.1", server_port=7860,share=True)
