from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from vosk import Model, KaldiRecognizer
from pathlib import Path
import pandas as pd
import json
import random
import re
import time

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
# LOAD DATASET
# =====================================================

df = pd.read_csv("explicit_words.csv")

NSFW_TERMS = set(
    df[df["explicit"] == 1]["term"].astype(str).str.lower()
)

print("Loaded NSFW terms:", len(NSFW_TERMS))

# =====================================================
# TEXT REPLACEMENT SYSTEM
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


def choose_replacement(length):
    if length <= 4:
        return random.choice(EMOJIS)
    elif length <= 7:
        return random.choice(FACTS)
    elif length <= 12:
        return random.choice(MATH)
    else:
        return random.choice(RIDDLES)


def normalize(word: str):

    word = word.lower()

    # -----------------------------
    # 1. Symbol → letter mapping
    # -----------------------------
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

    # -----------------------------
    # 2. Remove non-alphanumeric
    # -----------------------------
    word = re.sub(r"[^a-z0-9]", "", word)

    # -----------------------------
    # 3. Compress repeated letters
    # fuuuuuck → fuuck
    # -----------------------------
    word = re.sub(r"(.)\1{2,}", r"\1\1", word)

    return word



def moderate_text(text: str):

    clean_text = re.sub(r"[^a-z0-9\s]", " ", text.lower())

    # 🔴 If ANY explicit word exists → replace whole sentence
    for term in NSFW_TERMS:
        if term in clean_text:
            return choose_replacement(len(term))

    # otherwise keep original
    return text


# =====================================================
# TEXT ENDPOINT (Browser Extension Uses This)
# =====================================================

from pydantic import BaseModel

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
# LOAD VOSK MODEL
# =====================================================

print("Loading Vosk model...")
vosk_model = Model("vosk-model-small-en-us-0.15")
print("✅ Vosk ready")

recognizer = None
audio_clock_start = time.time()

# =====================================================
# EXPLICIT WORD DETECTION
# =====================================================

def detect_explicit(words_info):

    detected = []
    mute_segments = []

    for w in words_info:

        spoken = normalize(w.get("word", ""))

        if spoken in NSFW_TERMS:

            start = float(w.get("start", 0))
            end = float(w.get("end", 0))

            print(
                f"✅ FINAL DETECTED → {spoken} | {start:.2f}s → {end:.2f}s"
            )

            detected.append(spoken)

            mute_segments.append({
                "word": spoken,
                "start": round(start, 2),
                "end": round(end, 2)
            })

    return detected, mute_segments


# =====================================================
# AUDIO STREAM ENDPOINT (REAL-TIME SAFE)
# =====================================================

@app.post("/audio")
async def receive_audio(request: Request):

    global recognizer, audio_clock_start

    try:
        raw_audio = await request.body()

        # ---------- EMPTY CHECK ----------
        if not raw_audio:
            return {"text": "", "mute": [], "explicit_words": []}

        # ---------- ALIGN BUFFER (CRITICAL FIX) ----------
        if len(raw_audio) % 2 != 0:
            raw_audio = raw_audio[:-1]

        sample_rate = int(
            request.headers.get("X-Sample-Rate", 16000)
        )

        # ---------- INIT ONCE ----------
        if recognizer is None:
            recognizer = KaldiRecognizer(vosk_model, sample_rate)
            recognizer.SetWords(True)
            audio_clock_start = time.time()

        # ---------- STREAM AUDIO ----------
        recognizer.AcceptWaveform(raw_audio)

        # ---------- LIVE PARTIAL ----------
        partial_json = json.loads(recognizer.PartialResult())
        live_text = partial_json.get("partial", "")

        if live_text:
            print("LIVE:", live_text)

        # ---------- EARLY DETECTION ----------
        early_detected = []

        if live_text:
            approx_time = time.time() - audio_clock_start

            for w in live_text.split():
                clean = normalize(w)

                if clean in NSFW_TERMS:
                    early_detected.append(clean)

                    print(
                        f"⚡ EARLY DETECTED → {clean} | ~{approx_time:.2f}s"
                    )

        # ---------- FINAL RESULT ----------
        result_json = json.loads(recognizer.Result())

        text = result_json.get("text", "")
        words_info = result_json.get("result", [])

        detected = []
        mute_segments = []

        if words_info:
            detected, mute_segments = detect_explicit(words_info)

        all_detected = list(set(detected + early_detected))

        return {
            "text": live_text if live_text else text,
            "mute": mute_segments,
            "explicit_words": all_detected,
        }

    except Exception as e:
        print("Audio processing error:", e)

        return {
            "text": "",
            "mute": [],
            "explicit_words": []
        }
