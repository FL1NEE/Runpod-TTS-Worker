# -*- coding: utf-8 -*-
from typing import Optional, Any
import runpod
import os
import json
import base64
import tempfile
import traceback
import numpy as np
import boto3
from botocore.config import Config as BotoConfig
from io import BytesIO
from kokoro import KPipeline
import soundfile as sf
from pydub import AudioSegment

_s3_endpoint: Optional[str] = os.environ.get("BUCKET_ENDPOINT_URL")
_s3_access_key: Optional[str] = os.environ.get("BUCKET_ACCESS_KEY_ID")
_s3_secret_key: Optional[str] = os.environ.get("BUCKET_SECRET_ACCESS_KEY")
S3_BUCKET_NAME: str = os.environ.get("BUCKET_NAME", "aicaller")
S3_PUBLIC_URL: str = os.environ.get("BUCKET_PUBLIC_URL", "").rstrip("/")

s3_client: Optional[Any] = None
if _s3_endpoint and _s3_access_key and _s3_secret_key:
    s3_client = boto3.client(
        "s3",
        endpoint_url=_s3_endpoint,
        aws_access_key_id=_s3_access_key,
        aws_secret_access_key=_s3_secret_key,
        config=BotoConfig(signature_version="s3v4"),
    )
    print(f"worker-tts - S3 initialized (endpoint: {_s3_endpoint}, bucket: {S3_BUCKET_NAME})")
else:
    print("worker-tts - S3 not configured, will return base64")

# Voice presets: voice_id → (lang_code, kokoro_voice)
VOICE_PRESETS: dict = {
    "female_1": ("a", "af_heart"),
    "female_2": ("a", "af_bella"),
    "female_3": ("a", "af_nicole"),
    "female_uk": ("b", "bf_emma"),
    "male_1":   ("a", "am_adam"),
    "male_2":   ("a", "am_michael"),
    "male_uk":  ("b", "bm_george"),
}
DEFAULT_VOICE: str = "female_1"

# Lang code map: ISO language → Kokoro lang_code
LANG_MAP: dict = {
    "en": "a",
    "en-gb": "b",
    "es": "e",
    "fr": "f",
    "hi": "h",
    "it": "i",
    "pt": "p",
    "ja": "j",
    "zh": "z",
    "ko": "k",
}

print("worker-tts - Initializing Kokoro pipelines...")
_pipelines: dict[str, KPipeline] = {}
for lang_code in set(LANG_MAP.values()):
    try:
        _pipelines[lang_code] = KPipeline(lang_code=lang_code)
        print(f"worker-tts - Pipeline ready: lang_code={lang_code}")
    except Exception as e:
        print(f"worker-tts - Pipeline failed for lang_code={lang_code}: {e}")
print("worker-tts - Kokoro ready")


def validate_input(job_input: Any) -> tuple[Optional[dict], Optional[str]]:
    if job_input is None:
        return None, "Please provide input"

    if isinstance(job_input, str):
        try:
            job_input = json.loads(job_input)
        except json.JSONDecodeError:
            return None, "Invalid JSON format in input"

    text: Optional[str] = job_input.get("text")
    if not text or not isinstance(text, str):
        return None, "Missing or invalid 'text' parameter"
    if len(text) > 5000:
        return None, "'text' exceeds 5000 character limit"

    language: str = job_input.get("language", "en")
    if language not in LANG_MAP:
        return None, f"Unsupported language '{language}'. Supported: {sorted(LANG_MAP.keys())}"

    voice: str = job_input.get("voice", DEFAULT_VOICE)
    if voice not in VOICE_PRESETS:
        return None, f"Unknown voice '{voice}'. Available: {sorted(VOICE_PRESETS.keys())}"

    speed: float = float(job_input.get("speed", 1.0))
    if not (0.5 <= speed <= 2.0):
        return None, "'speed' must be between 0.5 and 2.0"

    return {
        "text": text,
        "language": language,
        "voice": voice,
        "speed": speed,
    }, None


def generate_speech(text: str, language: str, voice: str, speed: float) -> bytes:
    lang_code = LANG_MAP[language]
    _, kokoro_voice = VOICE_PRESETS[voice]

    # If voice preset has its own lang_code, use that
    preset_lang_code, _ = VOICE_PRESETS[voice]
    pipeline = _pipelines.get(lang_code) or _pipelines.get(preset_lang_code)
    if pipeline is None:
        raise RuntimeError(f"No pipeline for language '{language}'")

    audio_chunks = []
    for _, _, audio in pipeline(text, voice=kokoro_voice, speed=speed):
        if audio is not None:
            audio_chunks.append(audio)

    if not audio_chunks:
        raise RuntimeError("Kokoro returned no audio")

    audio_np = np.concatenate(audio_chunks)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
        wav_path = tmp_wav.name

    try:
        sf.write(wav_path, audio_np, 24000)
        segment = AudioSegment.from_wav(wav_path)
        mp3_buffer = BytesIO()
        segment.export(mp3_buffer, format="mp3", bitrate="128k")
        return mp3_buffer.getvalue()
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)


def upload_audio_to_s3(mp3_bytes: bytes, job_id: str) -> str:
    s3_key = f"outputs/{job_id}/output.mp3"
    s3_client.put_object(
        Bucket=S3_BUCKET_NAME,
        Key=s3_key,
        Body=mp3_bytes,
        ContentType="audio/mpeg",
        ACL="public-read",
    )
    base_url = S3_PUBLIC_URL or _s3_endpoint.rstrip("/")
    public_url = f"{base_url}/{S3_BUCKET_NAME}/{s3_key}"
    print(f"worker-tts - Uploaded: {public_url}")
    return public_url


def handler(job: dict) -> dict:
    job_id: str = job.get("id", "UNKNOWN")
    print(f"worker-tts - handler called: job_id={job_id}", flush=True)

    job_input: Any = job.get("input")
    validated, error = validate_input(job_input)
    if error:
        return {"error": error}

    try:
        print(f"worker-tts - Generating: voice={validated['voice']}, lang={validated['language']}, speed={validated['speed']}, chars={len(validated['text'])}")
        mp3_bytes = generate_speech(
            text=validated["text"],
            language=validated["language"],
            voice=validated["voice"],
            speed=validated["speed"],
        )
        print(f"worker-tts - Generated {len(mp3_bytes)} bytes")

        if s3_client:
            public_url = upload_audio_to_s3(mp3_bytes, job_id)
            return {
                "audio": {
                    "filename": "output.mp3",
                    "type": "url",
                    "data": public_url,
                }
            }

        return {
            "audio": {
                "filename": "output.mp3",
                "type": "base64",
                "data": base64.b64encode(mp3_bytes).decode("utf-8"),
            }
        }

    except Exception as e:
        print(f"worker-tts - Error: {e}\n{traceback.format_exc()}")
        return {"error": str(e)}


if __name__ == "__main__":
    print("worker-tts - Starting handler...")
    runpod.serverless.start({"handler": handler})
