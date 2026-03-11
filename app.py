# import os
# import torch
# import random
# import requests
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from PIL import Image
# from io import BytesIO
# from dotenv import load_dotenv
# from transformers import SiglipImageProcessor, SiglipForImageClassification

# load_dotenv()

# app = Flask(__name__)
# CORS(app)

# # Load model and processor
# MODEL_PATH = os.getenv('MODEL_PATH', 'models/siglip_nsfw/final')
# BASE_MODEL = "google/siglip-base-patch16-256"
# device = "cpu"  # Railway uses CPU

# print(f"Loading model from {MODEL_PATH}...")
# processor = SiglipImageProcessor.from_pretrained(BASE_MODEL)
# model = SiglipForImageClassification.from_pretrained(MODEL_PATH).to(device)
# model.eval()

# # Load facts
# with open('facts.txt', 'r') as f:
#     facts = [line.strip() for line in f if line.strip()]

# LABEL_MAP = {
#     0: "Anime Picture",
#     1: "Hentai",
#     2: "Normal",
#     3: "Pornography",
#     4: "Enticing or Sensual"
# }

# @app.route('/classify', methods=['POST'])
# def classify_image():
#     try:
#         data = request.json
#         image_url = data.get('imageUrl')
        
#         if not image_url:
#             return jsonify({'error': 'No image URL provided'}), 400
        
#         # Download image from URL
#         response = requests.get(image_url, timeout=5)
#         response.raise_for_status()
#         img = Image.open(BytesIO(response.content)).convert('RGB')
        
#         # Process and classify
#         inputs = processor(images=img, return_tensors="pt").to(device)
#         with torch.no_grad():
#             outputs = model(**inputs)
        
#         logits = outputs.logits
#         predicted_class = logits.argmax(-1).item()
#         confidence = float(torch.softmax(logits, dim=-1).max().item())
        
#         # Check if NSFW (class 2 is "Normal")
#         is_nsfw = predicted_class != 2
#         random_fact = random.choice(facts)
#         label = LABEL_MAP.get(predicted_class, "Unknown")
        
#         return jsonify({
#             'isNSFW': is_nsfw,
#             'fact': random_fact,
#             'confidence': confidence,
#             'label': label
#         })
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/health', methods=['GET'])
# def health():
#     return jsonify({'status': 'ok'})

# if __name__ == '__main__':
#     port = int(os.environ.get('PORT', 5000))
#     app.run(debug=False, port=port, host='0.0.0.0')


import os
import torch
import random
import requests
import json
import re
import time
from pathlib import Path
from io import BytesIO

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from PIL import Image
import pandas as pd

from vosk import Model, KaldiRecognizer

from transformers import SiglipImageProcessor, SiglipForImageClassification


# =====================================================
# FASTAPI SETUP
# =====================================================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# LOAD IMAGE MODEL (SigLIP)
# =====================================================

MODEL_PATH = os.getenv('MODEL_PATH', 'models/siglip_nsfw/final')
BASE_MODEL = "google/siglip-base-patch16-256"
device = "cpu"

print("Loading Image Model...")

processor = SiglipImageProcessor.from_pretrained(BASE_MODEL)
model = SiglipForImageClassification.from_pretrained(MODEL_PATH).to(device)
model.eval()

LABEL_MAP = {
    0: "Anime Picture",
    1: "Hentai",
    2: "Normal",
    3: "Pornography",
    4: "Enticing or Sensual"
}

# =====================================================
# LOAD DATASET (TEXT MODERATION)
# =====================================================

df = pd.read_csv("explicit_words.csv")

NSFW_TERMS = set(
    df[df["explicit"] == 1]["term"].astype(str).str.lower()
)

print("Loaded NSFW terms:", len(NSFW_TERMS))

# =====================================================
# LOAD REPLACEMENTS
# =====================================================

BASE = Path("replacements")

def load_lines(path):
    return [
        l.strip()
        for l in path.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]

EMOJIS = load_lines(BASE / "emojis.txt")
FACTS = load_lines(BASE / "facts.txt")
MATH = load_lines(BASE / "math.txt")
RIDDLES = load_lines(BASE / "riddles.txt")

# =====================================================
# NORMALIZATION
# =====================================================

def normalize(word: str):

    word = word.lower()

    substitutions = {
        "@": "a",
        "4": "a",
        "0": "o",
        "1": "i",
        "!": "i",
        "$": "s",
        "3": "e",
        "7": "t",
        "+": "t"
    }

    for k, v in substitutions.items():
        word = word.replace(k, v)

    word = re.sub(r"[^a-z0-9]", "", word)

    word = re.sub(r"(.)\1{2,}", r"\1\1", word)

    return word

# =====================================================
# REPLACEMENT SELECTION
# =====================================================

def choose_replacement():

    pool = EMOJIS + FACTS + MATH + RIDDLES
    return random.choice(pool)

# =====================================================
# TEXT MODERATION
# =====================================================

def moderate_text(text: str):

    clean_text = re.sub(r"[^a-z0-9\s]", " ", text.lower())

    for term in NSFW_TERMS:
        if term in clean_text:
            return choose_replacement()

    return text

# =====================================================
# TEXT API
# =====================================================

class TextRequest(BaseModel):
    text: str

@app.post("/text")
async def text_moderation(data: TextRequest):

    moderated = moderate_text(data.text)

    return {
        "original": data.text,
        "moderated": moderated
    }

# =====================================================
# LOAD VOSK MODEL (AUDIO)
# =====================================================

print("Loading Vosk model...")

vosk_model = Model("vosk-model-small-en-us-0.15")

print("Vosk ready")

recognizer = None
audio_clock_start = time.time()

# =====================================================
# AUDIO DETECTION
# =====================================================

def detect_explicit(words_info):

    detected = []
    mute_segments = []

    for w in words_info:

        spoken = normalize(w.get("word", ""))

        if spoken in NSFW_TERMS:

            start = float(w.get("start", 0))
            end = float(w.get("end", 0))

            detected.append(spoken)

            mute_segments.append({
                "word": spoken,
                "start": round(start, 2),
                "end": round(end, 2)
            })

    return detected, mute_segments

# =====================================================
# AUDIO API
# =====================================================

@app.post("/audio")
async def receive_audio(request: Request):

    global recognizer, audio_clock_start

    try:

        raw_audio = await request.body()

        if not raw_audio:
            return {"text": "", "mute": [], "explicit_words": []}

        if len(raw_audio) % 2 != 0:
            raw_audio = raw_audio[:-1]

        sample_rate = int(request.headers.get("X-Sample-Rate", 16000))

        if recognizer is None:
            recognizer = KaldiRecognizer(vosk_model, sample_rate)
            recognizer.SetWords(True)
            audio_clock_start = time.time()

        recognizer.AcceptWaveform(raw_audio)

        partial_json = json.loads(recognizer.PartialResult())
        live_text = partial_json.get("partial", "")

        result_json = json.loads(recognizer.Result())

        text = result_json.get("text", "")
        words_info = result_json.get("result", [])

        detected = []
        mute_segments = []

        if words_info:
            detected, mute_segments = detect_explicit(words_info)

        return {
            "text": live_text if live_text else text,
            "mute": mute_segments,
            "explicit_words": detected
        }

    except Exception as e:

        print("Audio processing error:", e)

        return {
            "text": "",
            "mute": [],
            "explicit_words": []
        }

# =====================================================
# IMAGE CLASSIFICATION API
# =====================================================

@app.post("/image")
async def classify_image(request: Request):

    try:

        data = await request.json()
        image_url = data.get("imageUrl")

        if not image_url:
            return {"error": "No image URL provided"}

        response = requests.get(image_url, timeout=5)
        img = Image.open(BytesIO(response.content)).convert('RGB')

        inputs = processor(images=img, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        predicted_class = logits.argmax(-1).item()
        confidence = float(torch.softmax(logits, dim=-1).max().item())

        label = LABEL_MAP.get(predicted_class, "Unknown")

        is_nsfw = predicted_class != 2

        return {
            "isNSFW": is_nsfw,
            "confidence": confidence,
            "label": label
        }

    except Exception as e:

        return {"error": str(e)}

# =====================================================
# HEALTH CHECK
# =====================================================

@app.get("/health")
def health():

    return {"status": "ok"}
