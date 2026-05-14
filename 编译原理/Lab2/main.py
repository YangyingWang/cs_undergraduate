import gradio as gr
import re

keywords = {'IF': 11, 'THEN': 12, 'ELSE': 13, 'INT': 14, 'CHAR': 15, 'FOR': 16}
separators = {'"': 21, ';': 22}
operators = {'=': 31, '>=': 32, '==': 33, '+': 34, '/': 35, '%': 36, '++': 37}
constant = {}
identifier = {}

def read_file(file_path):
    if file_path is None:
        return ""
    with open(file_path.name, 'r', encoding='utf-8') as file:
        return file.read()

def preprocess(text):
    text = re.sub(r'//.*?$|/\*.*?\*/', '', text, flags=re.DOTALL | re.MULTILINE)  # 除注释

    text = re.sub(r'\s+', ' ', text)  # 去掉多余空格和换行符
    text = text.strip() # 去掉首尾多余空格
    return text

# 词法分析器
def lexer(text):
    constant_cnt = 0
    identifier_cnt = 0
    tokens = []  # 保存所有的词法单元
    i = 0
    n = len(text)
    
    while i < n:
        ch = text[i]

        if ch.isspace():
            i += 1
            continue
        
        # 识别关键字和标识符
        if ch.isalpha():
            flag = 0
            start = i
            while i < n and (text[i].isalpha() or text[i].isdigit()):
                if not text[i].isalpha():
                    flag = 1
                i += 1
            word = text[start:i]

            if flag == 1 or len(word) > 10:
                tokens.append((word,'err'))
            elif word.upper() in keywords:
                tokens.append((word, keywords[word.upper()]))
            else:
                if word not in identifier:
                    identifier_cnt = identifier_cnt + 1
                    identifier[word] = 50 + identifier_cnt
                tokens.append((word,identifier[word]))
            continue
        
        # 识别常数
        elif ch.isdigit():
            flag = 0
            start = i
            while i < n and (text[i].isalpha() or text[i].isdigit()):
                if not text[i].isdigit():
                    flag = 1
                i += 1
            num = text[start:i]

            if flag == 1:
                tokens.append((num,'err'))
            else:
                if num not in constant:
                    constant_cnt = constant_cnt + 1
                    constant[num] = 40 + constant_cnt
                tokens.append((num, constant[num]))
            continue

        # 识别分隔符
        elif ch in separators:
            i += 1
            tokens.append((ch, separators[ch]))
            continue

        # 识别运算符
        else: 
            if ch in operators:
                if ch == '=' and i + 1 < n and text[i + 1] == '=':
                    i += 2
                    tokens.append(('==', operators['==']))
                elif ch == '+' and i + 1 < n and text[i + 1] == '+':
                    i += 2
                    tokens.append(('++', operators['++']))
                else:
                    i += 1
                    tokens.append((ch, operators[ch]))
            elif ch == '>' and i + 1 < n and text[i + 1] == '=':
                i += 2
                tokens.append(('>=', operators['>=']))
            elif ch == '<' and i + 1 < n and text[i + 1] == '=':
                i += 2
                tokens.append(('<=', 'err'))
            else:
                tokens.append((ch, 'err'))
                i += 1
    
    return tokens

# 主函数
def process_file(file):
    text = read_file(file)
    text = preprocess(text)
    tokens = lexer(text)

    results = ""
    for token in tokens:
        results += f"{token[0]}   ({token[0]}, {token[1]})\n"

    constant_table = list(constant.keys())
    identifier_table = list(identifier.keys())
    results += "\n常数表中的内容为： " + ",".join(constant_table)
    results += "\n变量表（标识符表）中的内容为： " + ",".join(identifier_table)
    return results

def clear_input():
    global constant, identifier
    constant = {}
    identifier = {}
    return None, "", ""

with gr.Blocks() as iface:
    with gr.Row():
        with gr.Column():
            file = gr.File(type="filepath", label="选择源文件")
            file_content = gr.Textbox(label="文件内容", interactive=False, lines=11)
        with gr.Column():
            content = gr.Textbox(label="词法分析结果", interactive=False, lines=25)
        
    with gr.Row():
        submit = gr.Button("提交检测")
        clear = gr.Button("clear")

    file.change(fn=lambda f: read_file(f), inputs=file, outputs=file_content)
    submit.click(fn=process_file, inputs=file, outputs=content)
    clear.click(fn=clear_input, inputs=[], outputs=[file, file_content, content])

iface.launch()
