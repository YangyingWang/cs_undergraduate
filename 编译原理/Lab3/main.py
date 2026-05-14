import gradio as gr

def read_file(file_path):
    if file_path is None:
        return ""
    with open(file_path.name, 'r', encoding='utf-8') as file:
        return file.read()

irrelevant_chars = {' ', ';', ','}

def lexer(text):
    state = 0
    i = 0
    while i < len(text):
        c = text[i]
        if c in irrelevant_chars:
            i += 1
            continue  # 跳过无关符号
        if state == 0:
            if c == 'a': state = 0
            elif c == 'b': state = 1
            else: return False

        elif state == 1:
            if c == 'a': state = 0
            elif c == 'b': state = 2
            else: return False

        elif state == 2:
            if c == 'a': state = 0
            elif c == 'b': state = 2
            elif c == '>': state = 3
            elif c == '<': state = 4
            elif c == '!': state = 5
            elif c == '=': state = 6
            else: return False

        elif state == 3:
            if c == '=': state = 7
            elif c == '1': state = 11
            else: return False

        elif state == 4:
            if c == '=': state = 8
            elif c == '1': state = 11
            else: return False

        elif state == 5:
            if c == '=': state = 9
            else: return False

        elif state == 6:
            if c == '=': state = 10
            else: return False

        elif state in [7,8,9,10]:
            if c == '1': state = 11
            else: return False

        elif state == 11:
            return i == len(text)-1

        i += 1
    return state == 11


def process_file(file):
    text = read_file(file)
    lines = text.splitlines()
    
    results = ""
    for line in lines:
        result = lexer(line)
        results += f"{line}      => {result}" + "\n"

    return results

def clear_input():
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
