import gradio as gr
import re
from collections import Counter

# 文件读取
def read_file(file_path):
    with open(file_path.name, 'r', encoding='utf-8') as file:
        return file.read()

# 预处理函数
def preprocess(text):
    text = re.sub(r'//.*?$|/\*.*?\*/|#.*?$', '', text, flags=re.DOTALL | re.MULTILINE)  # 除注释
    
    text = re.sub(r'""".*?"""', '', text, flags=re.DOTALL)  # Python文档字符串
    text = re.sub(r"'''.*?'''", '', text, flags=re.DOTALL)  # Python文档字符串
    
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)  # HTML注释

    text = re.sub(r'\s+', ' ', text)  # 去掉多余空格和换行符
    text = text.strip() # 去掉首尾多余空格
    text = text.lower() # 统一转换成小写
    return text

# 分词
def tokenize(text, n):
    words = re.findall(r'\b\w+\b|[^\w\s]', text)
    ngrams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    return words, ngrams

# 计算重复率的函数
def calculate_repetition_rate(test1, test2, n):
    words1, ngrams1 = tokenize(test1, n)
    words2, ngrams2 = tokenize(test2, n)

    set1 = set(ngrams1)
    set2 = set(ngrams2)

    # 计算交集和并集
    intersection = set1.intersection(set2)
    union = set1.union(set2)

    repetition_rate = len(intersection) / len(union) if len(union) > 0 else 0
    return repetition_rate*100, ngrams1, ngrams2

# 主函数
def process_files(file1, file2, n):
    content1 = read_file(file1)
    content2 = read_file(file2)

    preprocessed1 = preprocess(content1)
    preprocessed2 = preprocess(content2)

    repetition_rate, ngrams1, ngrams2 = calculate_repetition_rate(preprocessed1, preprocessed2, int(n))

    repeated_ngrams = set(ngrams1) & set(ngrams2)
    content3 = "\n".join([f"{' '.join(ngram)}" for ngram in repeated_ngrams])

    return f"{repetition_rate:.2f}%", content1, content2, content3

def clear_input():
    return None, None, "", "", "", ""  # 清空所有文本框内容

# 使用 Gradio 创建 UI
with gr.Blocks() as iface:
    with gr.Row():
        with gr.Column():
            file1 = gr.File(type="filepath", label="导入第一个文件")
        with gr.Column():
            file2 = gr.File(type="filepath", label="导入第二个文件")

    with gr.Row():
        # n = gr.Textbox(label="n（连续词数）", value="3")
        n = gr.Slider(minimum=1, maximum=30, value=3, step=1, label="n（连续词数）")
        
    with gr.Row():
        submit = gr.Button("提交检测")
        clear = gr.Button("clear")

    with gr.Row():
        with gr.Column():
            content1 = gr.Textbox(label="第一个文件内容", interactive=False)
        with gr.Column():
            content2 = gr.Textbox(label="第二个文件内容", interactive=False)

    with gr.Row():
        answer = gr.Textbox(label="重复率", interactive=False)
    
    with gr.Row():
        content3 = gr.Textbox(label="重复词摘录", interactive=False)

    submit.click(fn=process_files, inputs=[file1, file2, n], outputs=[answer, content1, content2, content3])
    clear.click(fn=clear_input, inputs=[], outputs=[file1, file2, answer, content1, content2, content3])

# 启动界面
iface.launch()
