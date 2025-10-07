# Labubu Sesli Asistan

Labubu, mikrofon üzerinden konuşmaları dinleyip Ollama modeliyle cevap veren küçük bir asistan betiği.

## Özellikler
- Persona dosyasından karakter ayarı
- Çevrim içi/çevrim dışı kontrolü
- Google Speech Recognition tabanlı konuşmadan metne dönüşüm (sonraki sürümlerde eklenecek)
- Ollama API ile etkileşim (sonraki sürümlerde eklenecek)
- macOS `say` ile sesli dönüş (sonraki sürümlerde eklenecek)

## Gereksinimler
Sanal ortam kullanmayı öneririm.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Çalıştırma
Kod commit planına göre kademeli olarak ilerliyor. Final sürümünde:

```bash
python sohbet.py
```

## Komut Akışı
- Gün 1: Temel ayarlar
- Gün 2: Ses kayıt yardımcıları
- Gün 3: Konuşma metne çevirme yardımcıları
- Gün 4: LLM bağlantısı
- Gün 5: Seslendirme ve ana döngü
