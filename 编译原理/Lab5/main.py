import re
import gradio as gr

def tokenize(line):
        """词法分析器"""
        tokens = []
        token_specs = [
            ('type', r'\b(int|float)\b'),                # 明确匹配类型关键字
            ('relop', r'(<=|>=|==|!=|<|>)'),             # 比较运算符
            ('id', r'[a-zA-Z_]\w*'),                     # 标识符
            ('num', r'\d+(\.\d+)?'),                     # 数字常量
            ('operator', r'(=|\+|-|\*|/|%)'),            # 算术/赋值符号
            ('paren', r'(;|\(|\)|,)'),                   # 分号、括号、逗号
            ('skip', r'[ \t\n]+'),                       # 空格
            ('err', r'.')                                # 错误字符
        ]
        # 创建正则表达式，用于匹配输入文本中的每种类型的标记（token）
        tok_regex = '|'.join(f'(?P<{name}>{regex})' for name, regex in token_specs)
        for mo in re.finditer(tok_regex, line):
            kind = mo.lastgroup
            value = mo.group()
            if kind == 'skip':
                continue
            elif kind == 'err':
                raise SyntaxError(f"非法字符: {value}")
            tokens.append((kind, value))
        tokens.append(('#','#'))  # 结束符号
        return tokens

class G3Parser:
    def __init__(self):
        self.symbol_table = []
        self.next_address = 100
        self.pos = 0
        self.tokens = []

    def reset(self):
        self.symbol_table.clear()
        self.next_address = 100
        self.pos = 0
    
    def consume(self, expected_type):
        if self.pos >= len(self.tokens):
            raise SyntaxError("意外的文件结尾")
        
        type, value = self.tokens[self.pos]
        if type != expected_type:
            return ""
        self.pos += 1
        return value
        
    def parse(self, tokens):
        self.tokens = tokens
        self.pos = 0
        self.D()
        return self.tokens[self.pos] == ('#', '#')

    def D(self):
        """D → TL"""
        L_in = self.T()
        self.L(L_in)

    def T(self):
        """T → int | float"""
        return self.consume('type')
        
    def L(self, L_in):
        """L → id R"""
        entry = self.consume('id')
        self.addType(entry, L_in)
        self.R(L_in)
    
    def R(self, L_in):
        """R → ,id R | ε"""
        while self.pos < len(self.tokens):
            if self.tokens[self.pos][1] == ',':
                self.consume('paren')
                entry = self.consume('id')
                self.addType(entry, L_in)
            elif self.tokens[self.pos][1] == ';':
                self.consume('paren')  # 允许分号结尾
                break
            else:
                break

    def addType(self, name, var_type):
        """语义动作：将变量加入符号表"""
        if name=="":
            return
        for entry in self.symbol_table:
            if entry['name'] == name and entry['type'] == var_type:
                return
        size = 4 if var_type == 'int' else 8
        self.symbol_table.append({'name': name, 'type': var_type, 'address': self.next_address})
        self.next_address += size

class G2Parser():
    def __init__(self):
        self.code = []
        self.tokens = []
        self.pos = 0
        self.temp_index = 1

    def new_temp(self):
        """生成一个新的临时变量名"""
        temp = f"t{self.temp_index}"
        self.temp_index += 1
        return temp
    
    def consume(self, expected_type=None):
        if self.pos >= len(self.tokens):
            raise SyntaxError("意外的文件结尾")
        
        t = self.tokens[self.pos]
        if expected_type and t[0] != expected_type:
            return False
        self.pos += 1
        return t
    
    def parse(self, tokens):
        self.tokens = tokens
        self.pos = 0
        self.temp_index = 1
        self.code = []
        self.E()
        return self.tokens[self.pos] == ('#', '#')

    def E(self):
        """E → TE'"""
        left = self.T()
        return self.E_prime(left)

    def E_prime(self, left):
        """E' → =T"""
        if self.tokens[self.pos][1] == '=':
            self.consume()
            right = self.T()
            self.code.append(('=', right, None, left))
            return left
        return left
        
    def T(self):
        """T → FT'"""
        T_prime_in = self.F()
        return self.T_prime(T_prime_in)

    def T_prime(self, T_prime_in):
        """T' → *FT | /FT'| %FT'"""
        while self.tokens[self.pos][1] in ['*',  '/', '%']:
            op = self.consume()[1]
            right = self.F()
            temp = self.new_temp()
            self.code.append((op, T_prime_in, right, temp))
            T_prime_in = temp  # 新的左操作数
        return T_prime_in
    
    def F(self):
        """F → (E) | id"""
        if self.tokens[self.pos][1] == '(':
            self.consume()
            val = self.E()
            self.consume()
            return val
        elif self.tokens[self.pos][0] == 'id':
            return self.consume('id')[1]

class G1Parser():
    def __init__(self):
        self.code = []
        self.tokens = []
        self.pos = 0
        self.temp_index = 1

    def new_temp(self):
        """生成一个新的临时变量名"""
        temp = f"t{self.temp_index}"
        self.temp_index += 1
        return temp
    
    def consume(self, expected_type=None):
        if self.pos >= len(self.tokens):
            raise SyntaxError("意外的文件结尾")
        
        t = self.tokens[self.pos]
        if expected_type and t[0] != expected_type:
            return False
        self.pos += 1
        return t
    
    def parse(self, tokens):
        self.tokens = tokens
        self.pos = 0
        self.temp_index = 1
        self.code = []
        self.E()
        return self.tokens[self.pos] == ('#', '#')

    def E(self):
        """E → TE'"""
        left = self.T()
        return self.E_prime(left)

    def E_prime(self, inherited):
        """E' → =T"""
        if self.tokens[self.pos][1] == '=':
            self.consume()
            right = self.T()
            self.code.append(('=', right, None, inherited))
            return inherited
        return inherited
        
    def T(self):
        """T → FT'"""
        left = self.F()
        return self.T_prime(left)

    def T_prime(self, inherited):
        """T' → +FT | -FT'"""
        while self.tokens[self.pos][1] in ['+',  '-']:
            op = self.consume()[1]
            right = self.F()
            temp = self.new_temp()
            self.code.append((op, inherited, right, temp))
            inherited = temp  # 新的左操作数
        return inherited
    
    def F(self):
        """F → (E) | id"""
        if self.tokens[self.pos][1] == '(':
            self.consume()
            val = self.E()
            self.consume()
            return val
        elif self.tokens[self.pos][0] == 'id':
            return self.consume('id')[1]

declaration = G3Parser()
assignment = G2Parser()
assignment1 = G1Parser()

def analyze(line):
    tokens = tokenize(line)
    first_type, _ = tokens[0]

    if first_type == 'type':
        if declaration.parse(tokens):
            output = "该语句符合文法 G3\n更新符号表:\n"
            output += f"{'Name':<10}{'Type':<10}{'Address':<10}\n"
            for entry in declaration.symbol_table:
                output += f"{entry['name']:<10}{entry['type']:<10}{entry['address']:<10}\n"
            return output
        
    elif first_type == 'id':
        if assignment.parse(tokens):
            output = "该语句符合文法 G2\n三元式中间代码:\n"
            for i, line in enumerate(assignment.code, 1):
                if line[0] == '=':
                    output += f"({i}) {line[3]} = {line[1]}\n"
                else:
                    output += f"({i}) {line[3]} = {line[1]} {line[0]} {line[2]}\n"
            return output
        if assignment1.parse(tokens):
            output = "该语句符合文法 G1\n三元式中间代码:\n"
            for i, line in enumerate(assignment1.code, 1):
                if line[0] == '=':
                    output += f"({i}) {line[3]} = {line[1]}\n"
                else:
                    output += f"({i}) {line[3]} = {line[1]} {line[0]} {line[2]}\n"
            return output
        
    return "错误：不符合文法G1~G3\n"

with gr.Blocks(title="语法分析器", theme=gr.themes.Soft()) as iface:
    with gr.Row():
        with gr.Column():
            file = gr.File(type="filepath", label="选择源文件")
            file_content = gr.Textbox(label="文件内容", interactive=False, lines=11)
        with gr.Column():
            output = gr.Textbox(label="语义分析结果", interactive=False, lines=25)
    with gr.Row():
        submit = gr.Button("提交分析")
        clear = gr.Button("clear")

    def clear_input():
        declaration.reset()
        return None, "", ""

    def read_file(file_path):
        if not file_path:
            return ""
        with open(file_path.name, 'r', encoding='utf-8') as file:
            return file.read()
        
    def process_file(file_content):
        lines = file_content.splitlines()
        
        results = ""
        for line in lines:
            result = analyze(line)
            results += f"{line}\n==> {result}" + "\n"
        return results
    
    file.change(fn=lambda f: read_file(f), inputs=file, outputs=file_content)
    submit.click(fn=process_file, inputs=file_content, outputs=output)
    clear.click(fn=clear_input, inputs=[], outputs=[file, file_content, output])

iface.launch()