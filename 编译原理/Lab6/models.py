import os
import requests
from openai import OpenAI

api_key = os.getenv("OPENAI_API_KEY", "sk-9dQcGV9rsM4nUZ3MB50dqIqZKhG0l1yZlZQ3vclepVoEmnxy")
base_urls  = ["https://api.chatanywhere.tech/v1", "https://api.chatanywhere.com.cn/v1"]
client = OpenAI(api_key=api_key, base_url=base_urls[1])

def get_model_list():
    url = base_urls[0] + "/models"
    headers = {
        'Authorization': f'Bearer {api_key}',
        'User-Agent': 'Apifox/1.0.0 (https://apifox.com)'
        }
    response = requests.request("GET", url, headers=headers)
    data = response.json()['data']
    models = [model['id'] for model in data]
    print(models)

if __name__ == '__main__':
    get_model_list()
