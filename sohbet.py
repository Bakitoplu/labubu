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

# =====================
# SES KAYDI
# =====================
def record_until_silence(samplerate=RECORD_SAMPLERATE, frame_ms=20, max_wait_seconds=15,
                         vad_aggr=int(os.getenv("LABUBU_VAD", "2")),
                         end_silence_ms=2500, prepad_ms=300):
    """Konuşma bitince kaydı sonlandırır ve PCM(int16) döner. VAD varsa VAD, yoksa enerji eşiği kullanır."""
    blocksize = int(samplerate * frame_ms / 1000)
    dtype = 'int16'

    # Enerji eşiği için parametreler
    energy_thresh = 500  # kaba bir eşik; mikrofonuna göre ayarlanabilir
    silence_needed = int(end_silence_ms / frame_ms)
    prepad_frames = int(prepad_ms / frame_ms)

    vad = webrtcvad.Vad(vad_aggr)

    ring = []
    buffer = bytearray()
    collecting = False
    silence_count = 0

    def callback(indata, frames, time, status):
        nonlocal collecting, silence_count, buffer, ring
        # ses sürücüsü durumlarını sessize al
        # if status:
        #     pass
        frame = bytes(indata)
        ring.append(frame)
        if len(ring) > prepad_frames:
            ring.pop(0)

        is_speech = vad.is_speech(frame, samplerate)

        if is_speech:
            if not collecting:
                # konuşma başladı → ring içeriğini prepend et
                for f in ring:
                    buffer.extend(f)
                collecting = True
            buffer.extend(frame)
            silence_count = 0
        else:
            if collecting:
                buffer.extend(frame)
                silence_count += 1

    stream = sd.RawInputStream(samplerate=samplerate, channels=1, dtype=dtype,
                               blocksize=blocksize, callback=callback)

    with stream:
        total_ms = 0
        while True:
            sd.sleep(frame_ms)
            total_ms += frame_ms
            if collecting and silence_count >= silence_needed:
                break
            if total_ms >= max_wait_seconds * 1000:
                # zaman aşımı: varsa buffer'ı döndür
                break

    return bytes(buffer)

def record_audio(duration=DEFAULT_RECORD_SECONDS, samplerate=RECORD_SAMPLERATE):
    print("\n🎙️ Konuş ({} sn)…".format(duration))
    recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()
    print("✅ Kayıt bitti.")
    return np.squeeze(recording)

