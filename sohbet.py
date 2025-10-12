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

# =====================
# STT
# =====================
def speech_to_text(audio_data, samplerate=RECORD_SAMPLERATE):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        import soundfile as sf
        sf.write(tmpfile.name, audio_data, samplerate)
        try:
            with sr.AudioFile(tmpfile.name) as source:
                audio = sr_rec.record(source)
            try:
                text = sr_rec.recognize_google(audio, language="tr-TR")
            except (sr.UnknownValueError, sr.RequestError):
                text = ""
            return text
        finally:
            try:
                os.remove(tmpfile.name)
            except Exception:
                pass

def speech_to_text_from_pcm(pcm_bytes: bytes, samplerate=RECORD_SAMPLERATE):
    """PCM(int16)->WAV->STT"""
    if not pcm_bytes:
        return ""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        import wave
        with wave.open(tmpfile, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(samplerate)
            wf.writeframes(pcm_bytes)
        path = tmpfile.name
    try:
        with sr.AudioFile(path) as source:
            audio = sr_rec.record(source)
        try:
            text = sr_rec.recognize_google(audio, language="tr-TR")
        except (sr.UnknownValueError, sr.RequestError):
            text = ""
        return text
    finally:
        try:
            os.remove(path)
        except Exception:
            pass

# =====================
# OLLAMA – STREAMING CEVAP
# =====================
def chat_with_labubu_stream(prompt, system_hint="Kısa ve net cevap ver." ):
    """
    Ollama'dan akışlı (token token) cevap döndürür. Her satır/parça yield edilir.
    """
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": True,
        **({"system": system_prompt} if system_prompt else {})
    }
    try:
        with requests.post(OLLAMA_URL, json=payload, stream=True, timeout=60) as r:
            r.raise_for_status()
            for raw in r.iter_lines(decode_unicode=True):
                if not raw:
                    continue
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    # Bazı Ollama sürümlerinde doğrudan metin gelebilir
                    yield raw
                    continue
                token = data.get("response", "")
                if token:
                    yield token
                if data.get("done"):
                    break
    except requests.RequestException as e:
        if isinstance(e, requests.ConnectionError):
            print("[Bilgi] Ollama bağlı değil (localhost:11434). Cevap üretilemedi.")
        else:
            print("[Bilgi] Ollama isteği başarısız oldu.")
        return

def chat_with_labubu(prompt, system_hint="Kısa ve net cevap ver."):
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        **({"system": system_prompt} if system_prompt else {})
    }
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        return data.get("response", "")
    except requests.RequestException as e:
        if isinstance(e, requests.ConnectionError):
            print("[Bilgi] Ollama bağlı değil (localhost:11434).")
        else:
            print("[Bilgi] Ollama isteği başarısız oldu.")
        return ""

# =====================
# Basit TTS macOS 'say'
# =====================
def speak(text: str):
    """Basit TTS: macOS 'say' kullanır; başarısız olursa sessizce geçer."""
    try:
        subprocess.run(["say", text], check=False)
    except Exception:
        pass

# =====================
# ANA DÖNGÜ
# =====================
if __name__ == "__main__":
    print("🎤 Basit mod: Konuş, susunca kayıt bitecek. Cevaptan sonra tekrar dinleyecek. Çıkmak için 'çık' de veya CTRL+C.")
    print(f"[Mod] {'ONLINE' if ONLINE else 'OFFLINE'}  |  FORCE_OFFLINE={'1' if FORCE_OFFLINE else '0'}")
    try:
        while True:
            # 1) Konuşma bitene kadar kaydet
            pcm = record_until_silence()
            if not pcm or len(pcm) < 3200:  # ~100ms altını at
                continue

            # 2) STT
            text = speech_to_text_from_pcm(pcm)
            if not text:
                print("(Algılanan metin yok)")
                continue
            print("\nSen:", text)

            # 3) Çıkış komutu
            if text.strip().lower() in {"çık", "exit", "quit"}:
                print("Labubu: Görüşürüz 👋")
                break

            # 4) LLM
            reply = chat_with_labubu(text)
            if not reply:
                print("Labubu: (cevap alınamadı)")
                continue
            print("Labubu:", reply)

            # 5) TTS
            speak(reply)

    except KeyboardInterrupt:
        print("\nLabubu: Çıkılıyor…")