import re
import gradio as gr

def read_file(file_path):
    if file_path is None:
        return ""
    with open(file_path.name, 'r', encoding='utf-8') as file:
        return file.read()

def is_valid_expr(expr):
    try:
        tokens = tokenize(expr)
        # 简单检查条件表达式的结构合法性，必须包含 id relop id 这样的格式
        if 'relop' in tokens and tokens.count('id') >= 2:
            return True
    except Exception:
        return False
    return False

def preprocess(code):
    code = code.strip()
    # 正则表达式匹配 if-then-else 的结构（贪婪处理嵌套）
    pattern = re.compile(r'^\s*if\s*\((.*?)\)\s*(.*?)\s*else\s*(.*?)$', re.IGNORECASE)
    match = pattern.match(code)
    def replace(match):
        cond = match.group(1).strip()
        then_part = match.group(2).strip()
        else_part = match.group(3).strip() if match.group(3) else None

        cond_tokens = tokenize(cond)
        for i in range(len(cond_tokens) - 2):  # -2 是因为最后一个是 '#'
            if cond_tokens[i] == 'relop' and cond_tokens[i+1] == 'relop':
                return "invalid"
        if 'relop' not in cond_tokens or (cond_tokens.count('id') + cond_tokens.count('num')) < 2:
            return "invalid1"
        
        else_tokens = tokenize(else_part)
        if len(else_tokens) < 2:
            return "invalid3"

        # 对 then 和 else 中的 if 进行递归处理
        then_part = preprocess(then_part)
        if then_part == "invalid":
            return "invalid"
        if then_part.startswith("if string"):
            then_result = then_part
        else:
            then_tokens = tokenize(then_part)
            if len(then_tokens) < 2 or then_tokens[-2] != ';':
                return "invalid2"
            then_result = "string"

        if else_part:
            else_part = preprocess(else_part)
            if else_part == "invalid":
                return "invalid"
            if else_part.startswith("if string"):
                else_result = else_part
            else:
                else_tokens = tokenize(else_part)
                if len(else_tokens) < 2 :
                    return "invalid2"
                else_result = "string"
                return f"if string then {then_result} else {else_result}"
        else:
            return f"if string then {then_result}"

    match = pattern.match(code)
    if match:
        return replace(match)
    return code

def tokenize(code):
    if code.strip() == 'string':
        return ['string', '#']
    tokens = []
    token_specs = [
        ('relop', r'(<=|>=|==|!=|<|>)'),             # 比较运算符
        ('id', r'[a-zA-Z_]\w*'),                     # 标识符
        ('num', r'\d+(\.\d+)?'),                     # 数字常量
        ('operator', r'(=|\+|-|\*|/|%)'),            # 算术/赋值符号
        ('paren', r'(;|\(|\)|,)'),                   # 分号、括号、逗号
        ('skip', r'[ \t\n]+'),                       # 空格、制表符、换行符
        ('mismatch', r'.')                           # 匹配不到的字符
    ]
    # 创建正则表达式，用于匹配输入文本中的每种类型的标记（token）
    tok_regex = '|'.join(f'(?P<{name}>{regex})' for name, regex in token_specs)
    for match in re.finditer(tok_regex, code):
        kind = match.lastgroup     # 获取匹配到的标记类型
        value = match.group()      # 获取匹配到的具体值
        if kind == 'skip':
            continue
        elif kind == 'num':
            tokens.append(kind)
        elif kind == 'id' and value in {'if', 'then', 'else', 'int', 'float', 'mod', 'string'}:
            tokens.append(value)
        elif kind == 'id' or kind == 'relop':
            tokens.append(kind)
        elif kind == 'mismatch':
            raise RuntimeError(f"非法字符: {value}")
        else:
            tokens.append(value)
    tokens.append('#')  # 结束符号
    return tokens

G1_table = {
    'L': {'id': ['id', '=', 'E']},
    'E': {'id': ['F', "E'"], '(': ['F', "E'"]},
    "E'": {'+': ['+', 'F', "E'"], '-': ['-', 'F', "E'"], ')': [], '#': []},
    'F': {'id': ['id'], '(': ['(', 'E', ')']}
}

G2_table = {
    'L': {'id': ['id', '=', 'E']},
    'E': {'id': ['T', "E'"], '(': ['T', "E'"]},
    "E'": {'*': ['*', 'T', "E'"], ')': [], '#': []},
    'T': {'id': ['F', "T'"], '(': ['F', "T'"]},
    "T'": {'*': [], '/': ['/', 'F', "T'"], '%': ['%', 'F', "T'"], ')': [], '#': []},
    'F': {'id': ['id'], '(': ['(', 'E', ')']}
}

G3_table = {
    'D': {'int': ['T', 'L'], 'float': ['T', 'L']},
    'T': {'int': ['int'], 'float': ['float']},
    'L': {'id': ['id', 'R']},
    'R': {',': [',', 'id', 'R'], '#': []}
}

G4_table = {
    'L': {'id': ['E', ';', 'L'], 'num': ['E', ';', 'L'], '(': ['E', ';', 'L'], '#': []},
    'E': {'id': ['T', "E'"], 'num': ['T', "E'"], '(': ['T', "E'"]},
    "E'": {'+': ['+', 'T', "E'"], '-': ['-', 'T', "E'"], ')': [], ';': []},
    'T': {'id': ['F', "T'"], 'num': ['F', "T'"], '(': ['F', "T'"]},
    "T'": {'+': [], '-': [], '*': ['*', 'F', "T'"], '/': ['/', 'F', "T'"], 'mod': ['mod', 'F', "T'"], ')': [], ';': []},
    'F': {'id': ['id'], 'num': ['num'], '(': ['(', 'E', ')']}
}

G5_table = {
    'S': {'if': ['if', 'E', 'then', 'S', "S'"], 'string': ['E']},
    "S'": {'else': ['else', 'S'], '#': [], 'then': []},
    'E': {'string': ['string']}
}

def parse(tokens, table, start_symbol):
    stack = ['#', start_symbol]  # 初始化堆栈，包含结束符号（#）和起始符号
    i = 0

    while stack:
        top = stack.pop()
        current = tokens[i]
        if top == current:  # 如果堆栈顶部符号和当前符号相同
            i += 1  # 向后移动一位，处理下一个符号
        elif top in table:  # 如果堆栈顶部符号是非终结符（存在于预测分析表中）
            if current in table[top]:  # 如果当前输入符号可以从预测分析表中找到匹配的产生式
                rule = table[top][current]  # 根据当前符号找到对应的产生式
                for symbol in reversed(rule):  # 逆序处理产生式
                    if symbol != '':  # 只有非空符号才需要推入堆栈
                        stack.append(symbol)
            else:
                return False  # 如果没有找到匹配的产生式，报错
        else:
            return False  # 如果堆栈顶部符号是终结符且不匹配当前输入符号，报错
    return i == len(tokens)  # 如果所有输入符号都被成功匹配，返回True

def analyze(code):
    if code.startswith("if"):
        preprocessed = preprocess(code)
        print(preprocessed)
        tokens = tokenize(preprocessed)
        if parse(tokens, G5_table, 'S'):
            return "该语句符合文法 G5"
    else:
        tokens = tokenize(code)
        if parse(tokens, G1_table, 'L'):
            return "该语句符合文法 G1"
        elif parse(tokens, G2_table, 'L'):
            return "该语句符合文法 G2"
        elif parse(tokens, G3_table, 'D'):
            return "该语句符合文法 G3"
        elif parse(tokens, G4_table, 'L'):
            return "该语句符合文法 G4"
    return "错误：不符合任何已定义的文法（G1-G5）"

def process_file(file):
    text = read_file(file)
    lines = text.splitlines()
    
    results = ""
    for line in lines:
        result = analyze(line)
        results += f"{line}\t\t\t=> {result}" + "\n\n"

    return results

def clear_input():
    return None, "", ""

with gr.Blocks(theme=gr.themes.Ocean()) as iface:
    with gr.Row():
        with gr.Column():
            file = gr.File(type="filepath", label="选择源文件")
            file_content = gr.Textbox(label="文件内容", interactive=False, lines=11)
        with gr.Column():
            content = gr.Textbox(label="语法分析结果", interactive=False, lines=25)
        
    with gr.Row():
        submit = gr.Button("提交检测")
        clear = gr.Button("clear")

    file.change(fn=lambda f: read_file(f), inputs=file, outputs=file_content)
    submit.click(fn=process_file, inputs=file, outputs=content)
    clear.click(fn=clear_input, inputs=[], outputs=[file, file_content, content])

iface.launch()