# -*- coding: utf-8 -*-
from typing import Optional, Any
import runpod
import os
import json
import base64
import tempfile
import traceback
import torch
import numpy as np
import boto3
from botocore.config import Config as BotoConfig
from io import BytesIO
from TTS.api import TTS
from pydub import AudioSegment
import soundfile as sf

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

_s3_endpoint: Optional[str] = os.environ.get("BUCKET_ENDPOINT_URL")
_s3_access_key: Optional[str] = os.environ.get("BUCKET_ACCESS_KEY_ID")
_s3_secret_key: Optional[str] = os.environ.get("BUCKET_SECRET_ACCESS_KEY")
S3_BUCKET_NAME: str = os.environ.get("BUCKET_NAME", "tts-outputs")
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

print(f"worker-tts - Loading XTTS v2 model on {DEVICE}...")
_tts_model: TTS = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(DEVICE)
print("worker-tts - Model loaded")


SUPPORTED_LANGUAGES: set = {
    "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru",
    "nl", "cs", "ar", "zh-cn", "hu", "ko", "ja", "hi",
}


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
    if language not in SUPPORTED_LANGUAGES:
        return None, f"Unsupported language '{language}'. Supported: {sorted(SUPPORTED_LANGUAGES)}"

    speed: float = float(job_input.get("speed", 1.0))
    if not (0.5 <= speed <= 2.0):
        return None, "'speed' must be between 0.5 and 2.0"

    temperature: float = float(job_input.get("temperature", 0.75))
    if not (0.1 <= temperature <= 1.0):
        return None, "'temperature' must be between 0.1 and 1.0"

    speaker_wav_b64: Optional[str] = job_input.get("speaker_wav")

    return {
        "text": text,
        "language": language,
        "speed": speed,
        "temperature": temperature,
        "speaker_wav_b64": speaker_wav_b64,
    }, None


def generate_speech(
    text: str,
    language: str,
    speed: float,
    temperature: float,
    speaker_wav_path: Optional[str],
) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
        wav_path = tmp_wav.name

    try:
        if speaker_wav_path:
            _tts_model.tts_to_file(
                text=text,
                language=language,
                speaker_wav=speaker_wav_path,
                speed=speed,
                temperature=temperature,
                file_path=wav_path,
            )
        else:
            speakers = _tts_model.speakers or []
            speaker = speakers[0] if speakers else None
            _tts_model.tts_to_file(
                text=text,
                language=language,
                speaker=speaker,
                speed=speed,
                temperature=temperature,
                file_path=wav_path,
            )

        audio = AudioSegment.from_wav(wav_path)
        mp3_buffer = BytesIO()
        audio.export(mp3_buffer, format="mp3", bitrate="128k")
        return mp3_buffer.getvalue()
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)


def upload_audio_to_s3(mp3_bytes: bytes, job_id: str) -> str:
    if not s3_client:
        raise RuntimeError("S3 not configured")

    s3_key = f"outputs/{job_id}/output.mp3"
    s3_client.put_object(
        Bucket=S3_BUCKET_NAME,
        Key=s3_key,
        Body=mp3_bytes,
        ContentType="audio/mpeg",
        ACL="public-read",
    )
    print(f"worker-tts - Uploaded to S3: {s3_key}")

    base_url = S3_PUBLIC_URL or _s3_endpoint.rstrip("/")
    public_url = f"{base_url}/{S3_BUCKET_NAME}/{s3_key}"
    print(f"worker-tts - Public URL: {public_url}")
    return public_url


def handler(job: dict) -> dict:
    job_id: str = job.get("id", "UNKNOWN")
    print(f"worker-tts - handler called: job_id={job_id}", flush=True)

    job_input: Any = job.get("input")
    validated, error = validate_input(job_input)
    if error:
        return {"error": error}

    speaker_wav_path: Optional[str] = None
    tmp_speaker_path: Optional[str] = None

    try:
        if validated["speaker_wav_b64"]:
            raw = validated["speaker_wav_b64"]
            b64_data = raw.split(",", 1)[1] if "," in raw else raw
            speaker_bytes = base64.b64decode(b64_data)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(speaker_bytes)
                tmp_speaker_path = tmp.name
            speaker_wav_path = tmp_speaker_path
            print(f"worker-tts - Using custom speaker wav ({len(speaker_bytes)} bytes)")

        print(f"worker-tts - Generating speech: lang={validated['language']}, speed={validated['speed']}, len={len(validated['text'])}")
        mp3_bytes = generate_speech(
            text=validated["text"],
            language=validated["language"],
            speed=validated["speed"],
            temperature=validated["temperature"],
            speaker_wav_path=speaker_wav_path,
        )
        print(f"worker-tts - Generated {len(mp3_bytes)} bytes of MP3")

        if s3_client:
            public_url = upload_audio_to_s3(mp3_bytes, job_id)
            return {
                "audio": {
                    "filename": "output.mp3",
                    "type": "url",
                    "data": public_url,
                }
            }
        else:
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
    finally:
        if tmp_speaker_path and os.path.exists(tmp_speaker_path):
            os.remove(tmp_speaker_path)


if __name__ == "__main__":
    print("worker-tts - Starting handler...")
    runpod.serverless.start({"handler": handler})
