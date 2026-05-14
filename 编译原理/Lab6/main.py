import os
os.environ["PATH"] += os.pathsep + r"D:/ffmpeg-7.1.1-essentials_build/bin"
import gradio as gr
from openai import OpenAI
import whisper
import pyttsx3
import tempfile
import base64

whisper_model = whisper.load_model("base")
def audio_to_text(audio):
    result = whisper_model.transcribe(audio, word_timestamps=True, fp16=False, language='zh', task='transcribe')
    return result['text']

def text_to_audio(text):
    if not text.strip():
        return None
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 200)
        engine.setProperty("volume", 1.0)
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp_file.close()  # pyttsx3 会写入路径

        engine.save_to_file(text, tmp_file.name)
        engine.runAndWait()
        print(f"✅ 本地 TTS 生成成功: {tmp_file.name}")
        return tmp_file.name
    except Exception as e:
        print(f"❌ 本地 TTS 失败: {e}")
        return None

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

api_key = os.getenv("OPENAI_API_KEY", "sk-9dQcGV9rsM4nUZ3MB50dqIqZKhG0l1yZlZQ3vclepVoEmnxy")
base_urls  = ["https://api.chatanywhere.tech/v1", "https://api.chatanywhere.com.cn/v1"]
client = OpenAI(api_key=api_key, base_url=base_urls[1])

def clean_history(history):
    cleaned = []
    for msg in history:
        if isinstance(msg["content"], tuple) or msg["content"]=="":
            continue
        else:
            cleaned.append(msg)
    print(cleaned)
    return cleaned

def predict(message, history, select_model, t, p):
    history = clean_history(history)

    input = []
    image_list = []
    if message["text"] != '':
        input.append(message["text"])

    for file_path in message["files"]:
        if file_path.lower().endswith(".wav"):
            text = audio_to_text(file_path)
            if text:
                input.append(text)
        elif file_path.lower().endswith((".jpg", ".jpeg", ".png")):
            img = encode_image(file_path)       
            image_list.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img}"}
            })
    prompt = "\n".join(input).strip()
    if image_list:
        history.append({
            "role": "user","content": [
                {"type": "text", "text": prompt or "请分析这张图"},
                *image_list
            ]
        })
    else:
        if not prompt:
            response="⚠️ 请输入文本或语音"
            audio_path=text_to_audio(response)
            return response, audio_path
        history.append({"role": "user", "content": prompt})

    print(f"History after adding transcription: {history}")
    try:
        completion = client.chat.completions.create(
            model=select_model, 
            messages=history, 
            temperature=t,
            top_p=p
        )
        response=completion.choices[0].message.content
        audio_path=text_to_audio(response)
        return response, audio_path

    except Exception as e:
        response=f"⚠️ 错误: {str(e)}"
        audio_path=text_to_audio(response)
        return response, audio_path

with gr.Blocks(theme=gr.themes.Ocean()) as demo:
    audio = gr.Audio(label="语音回复", render=False, type="filepath", autoplay=True)
    chatbot=gr.ChatInterface(
        predict, 
        type="messages",
        title="聊天机器人",
        multimodal=True,
        textbox=gr.MultimodalTextbox(
            file_count="multiple", 
            placeholder="请输入任何想问的问题...",
            file_types=["image", "pdf", "txt", "audio"], 
            sources=["upload", "microphone"]
        ),
        additional_inputs=[
            gr.Radio(
                choices=["gpt-4o-mini", "gpt-4o-ca", "gpt-4o", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-4.1", "gpt-3.5-turbo", "gpt-3.5-turbo-ca", "deepseek-r1", "deepseek-v3"],
                label="选择模型",
                value="gpt-4o-mini"
            ),
            gr.Slider(label="temperature", value=0.4, minimum=0, maximum=1, step=0.1),
            gr.Slider(label="nucleus sampling", value=0.7, minimum=0, maximum=1, step=0.1)
        ],
        additional_outputs=[audio],
        save_history=True
    )
    with gr.Row():
        audio.render()
 
demo.launch()