# -*- coding: utf-8 -*-
from typing import Optional, Any
import runpod
import os
import json
import base64
import tempfile
import traceback
import requests
import boto3
from botocore.config import Config as BotoConfig
from faster_whisper import WhisperModel

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
    print(f"worker-stt - S3 initialized (endpoint: {_s3_endpoint}, bucket: {S3_BUCKET_NAME})")
else:
    print("worker-stt - S3 not configured, result will be returned inline")

WHISPER_MODEL_SIZE: str = os.environ.get("WHISPER_MODEL", "large-v3")
COMPUTE_TYPE: str = os.environ.get("WHISPER_COMPUTE_TYPE", "float16")

print(f"worker-stt - Loading Whisper {WHISPER_MODEL_SIZE} ({COMPUTE_TYPE})...")
_whisper_model: WhisperModel = WhisperModel(
    WHISPER_MODEL_SIZE,
    device="cuda",
    compute_type=COMPUTE_TYPE,
)
print("worker-stt - Model loaded")


def validate_input(job_input: Any) -> tuple[Optional[dict], Optional[str]]:
    if job_input is None:
        return None, "Please provide input"

    if isinstance(job_input, str):
        try:
            job_input = json.loads(job_input)
        except json.JSONDecodeError:
            return None, "Invalid JSON format in input"

    audio_b64: Optional[str] = job_input.get("audio")
    audio_url: Optional[str] = job_input.get("audio_url")

    if not audio_b64 and not audio_url:
        return None, "Missing 'audio' (base64) or 'audio_url' parameter"

    task: str = job_input.get("task", "transcribe")
    if task not in ("transcribe", "translate"):
        return None, "'task' must be 'transcribe' or 'translate'"

    return {
        "audio_b64": audio_b64,
        "audio_url": audio_url,
        "language": job_input.get("language"),
        "task": task,
        "word_timestamps": bool(job_input.get("word_timestamps", False)),
        "vad_filter": bool(job_input.get("vad_filter", True)),
    }, None


def transcribe(audio_path: str, language: Optional[str], task: str, word_timestamps: bool, vad_filter: bool) -> dict:
    segments_iter, info = _whisper_model.transcribe(
        audio_path,
        language=language,
        task=task,
        word_level_timestamps=word_timestamps,
        vad_filter=vad_filter,
    )

    segments = []
    full_text_parts = []

    for seg in segments_iter:
        seg_data = {
            "start": round(seg.start, 3),
            "end": round(seg.end, 3),
            "text": seg.text.strip(),
        }
        if word_timestamps and seg.words:
            seg_data["words"] = [
                {"start": round(w.start, 3), "end": round(w.end, 3), "word": w.word}
                for w in seg.words
            ]
        segments.append(seg_data)
        full_text_parts.append(seg.text.strip())

    return {
        "text": " ".join(full_text_parts),
        "language": info.language,
        "language_probability": round(info.language_probability, 4),
        "duration": round(info.duration, 3),
        "segments": segments,
    }


def upload_json_to_s3(result: dict, job_id: str) -> str:
    s3_key = f"outputs/{job_id}/result.json"
    body = json.dumps(result, ensure_ascii=False, indent=2).encode("utf-8")
    s3_client.put_object(
        Bucket=S3_BUCKET_NAME,
        Key=s3_key,
        Body=body,
        ContentType="application/json",
        ACL="public-read",
    )
    base_url = S3_PUBLIC_URL or _s3_endpoint.rstrip("/")
    public_url = f"{base_url}/{S3_BUCKET_NAME}/{s3_key}"
    print(f"worker-stt - JSON uploaded: {public_url}")
    return public_url


def handler(job: dict) -> dict:
    job_id: str = job.get("id", "UNKNOWN")
    print(f"worker-stt - handler called: job_id={job_id}", flush=True)

    job_input: Any = job.get("input")
    validated, error = validate_input(job_input)
    if error:
        return {"error": error}

    tmp_audio_path: Optional[str] = None

    try:
        if validated["audio_b64"]:
            raw = validated["audio_b64"]
            b64_data = raw.split(",", 1)[1] if "," in raw else raw
            audio_bytes = base64.b64decode(b64_data)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_audio_path = tmp.name
            print(f"worker-stt - Audio from base64 ({len(audio_bytes)} bytes)")

        elif validated["audio_url"]:
            resp = requests.get(validated["audio_url"], timeout=120)
            resp.raise_for_status()
            ext = os.path.splitext(validated["audio_url"].split("?")[0])[1] or ".wav"
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                tmp.write(resp.content)
                tmp_audio_path = tmp.name
            print(f"worker-stt - Audio from URL ({len(resp.content)} bytes)")

        print(f"worker-stt - Transcribing: task={validated['task']}, lang={validated['language']}, vad={validated['vad_filter']}")
        result = transcribe(
            audio_path=tmp_audio_path,
            language=validated["language"],
            task=validated["task"],
            word_timestamps=validated["word_timestamps"],
            vad_filter=validated["vad_filter"],
        )
        print(f"worker-stt - Done: lang={result['language']}, duration={result['duration']}s, segments={len(result['segments'])}")

        if s3_client:
            json_url = upload_json_to_s3(result, job_id)
            return {
                "result_url": json_url,
                "text": result["text"],
                "language": result["language"],
                "duration": result["duration"],
            }

        return result

    except Exception as e:
        print(f"worker-stt - Error: {e}\n{traceback.format_exc()}")
        return {"error": str(e)}
    finally:
        if tmp_audio_path and os.path.exists(tmp_audio_path):
            os.remove(tmp_audio_path)


if __name__ == "__main__":
    print("worker-stt - Starting handler...")
    runpod.serverless.start({"handler": handler})
