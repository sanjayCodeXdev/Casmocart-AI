import os
from dotenv import load_dotenv

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "google/gemini-2.0-flash-001"

# CAMERA SETTINGS
# 0 = Default Laptop Webcam
# 1 = External USB Camera
CAMERA_SOURCE = 0 

def call_openrouter(messages):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "AI For Her"
    }

    data = {
        "model": MODEL,
        "messages": messages
    }

    response = requests.post(BASE_URL, headers=headers, json=data)

    # print("DEBUG RESPONSE:", response.text)   # 👈 VERY IMPORTANT

    result = response.json()

    if "choices" not in result:
        raise Exception(f"OpenRouter Error: {result}")

    return result["choices"][0]["message"]["content"]