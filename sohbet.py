import requests
import subprocess
import sounddevice as sd
import numpy as np
import tempfile
import os
import json
import warnings
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

import socket
import webrtcvad

# Persona (Labubu karakteri)
PERSONA_FILE = os.getenv("LABUBU_PERSONA", "labubu_persona.txt")
system_prompt = ""
if os.path.exists(PERSONA_FILE):
    try:
        with open(PERSONA_FILE, "r", encoding="utf-8") as f:
            system_prompt = f.read()
    except Exception:
        system_prompt = ""

def _is_online(timeout=1.0) -> bool:
    """Basit internet kontrolü: 8.8.8.8:53'e TCP denemesi (DNS)."""
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=timeout)
        return True
    except OSError:
        return False

# Çalışma anında çevrimdışı/çevrimiçi kararını ver

FORCE_OFFLINE = os.getenv("LABUBU_FORCE_OFFLINE", "0") == "1"
ONLINE = (not FORCE_OFFLINE) and _is_online()

# STT: yalnizca openai-whisper kullan
# import whisper as OWWhisper  # (offline STT) — KAPALI (internet giderse aç)
import speech_recognition as sr  # (online STT)

# =====================
# CONFIG
# =====================
OLLAMA_URL = "http://localhost:11434/api/generate"
MARYTTS_URL = "http://localhost:59125/process"
MODEL_NAME = os.getenv("LABUBU_LLM", "qwen2.5:3b-instruct")  # Eski varsayılan model
WHISPER_MODEL = os.getenv("LABUBU_STT", "base")  # "tiny" > "base" > "small" hız/ses kalitesi dengesi
RECORD_SAMPLERATE = 16000
DEFAULT_RECORD_SECONDS = 1  # Daha hızlı dönüş için 1 sn
SENTENCE_PUNCT = ".!?…"  # Cümle sonu karakterleri

# =====================
# MODELLER (lazy load)
# =====================
# Online: speech_recognition; Offline: Whisper varsa onu kullan
stt_model = None
sr_rec = sr.Recognizer()

