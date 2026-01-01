"""
OpenAI-Compatible TTS Streaming Server using VoxCPM.

Provides:
- POST /v1/audio/speech - OpenAI-compatible TTS endpoint with streaming
- GET / - Serves the HTML frontend
"""

import asyncio
import io
import struct
import time
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, Generator

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from voxcpm import VoxCPM

# Enable TensorFloat32 for better GPU performance
torch.set_float32_matmul_precision('high')


# Global model instance
model: Optional[VoxCPM] = None
# Thread pool for TTS generation (single worker to avoid GPU contention)
executor = ThreadPoolExecutor(max_workers=1)


class TTSRequest(BaseModel):
    """OpenAI-compatible TTS request."""
    model: str = Field(
        default="tts-1", description="Model ID (ignored, uses VoxCPM)")
    input: str = Field(..., description="Text to synthesize")
    voice: str = Field(default="default",
                       description="Voice name (for future prompt support)")
    response_format: str = Field(
        default="wav", description="Audio format: wav or pcm")
    speed: float = Field(default=1.0, ge=0.25, le=4.0,
                         description="Speed factor (affects quality/speed tradeoff)")


def create_wav_header(sample_rate: int, num_channels: int = 1, bits_per_sample: int = 16, data_size: int = 0) -> bytes:
    """Create a WAV file header."""
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8

    # Use 0xFFFFFFFF for streaming (unknown size)
    if data_size == 0:
        data_size = 0xFFFFFFFF - 36

    header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF',
        data_size + 36,  # File size - 8
        b'WAVE',
        b'fmt ',
        16,  # Subchunk1 size (PCM)
        1,   # Audio format (1 = PCM)
        num_channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b'data',
        data_size
    )
    return header


def float32_to_int16(audio: np.ndarray) -> bytes:
    """Convert float32 audio [-1, 1] to int16 bytes."""
    # Clip to valid range and convert
    audio = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio * 32767).astype(np.int16)
    return audio_int16.tobytes()


def generate_tts_chunks(
    text: str,
    inference_timesteps: int,
    sample_rate: int,
    is_wav: bool,
    chunk_queue: queue.Queue,
):
    """Generate TTS audio chunks and put them in a queue. Runs in a thread."""
    generation_start = time.time()
    first_byte_time = None
    total_samples = 0
    first_chunk = True

    try:
        for chunk in model.generate_streaming(
            text=text,
            prompt_wav_path="/home/serj/tmp/example2.wav",
            prompt_text="All right, so have you ever heard of a little thing named text to speech? Well, it allows you to convert text into speech. I know that's super cool, isn't it?",
            cfg_value=2.0,
            # LocDiT inference timesteps, higher for better result, lower for fast speed
            inference_timesteps=5,
            # enable external TN tool, but will disable native raw text support
            normalize=False,
            # enable external Denoise tool, but it may cause some distortion and restrict the sampling rate to 16kHz
            denoise=False,
            # enable retrying mode for some bad cases (unstoppable)
            retry_badcase=True,
            retry_badcase_max_times=3,  # maximum retrying times
            retry_badcase_ratio_threshold=6.0,
        ):
            if first_chunk:
                first_byte_time = time.time() - generation_start
                print(f"[TTS] First chunk in {first_byte_time*1000:.0f}ms")

                # Send WAV header first
                if is_wav:
                    chunk_queue.put(create_wav_header(sample_rate))

                first_chunk = False

            # Convert to int16 and queue
            audio_bytes = float32_to_int16(chunk)
            total_samples += len(chunk)
            chunk_queue.put(audio_bytes)

        # Log performance metrics
        total_time = time.time() - generation_start
        audio_duration = total_samples / sample_rate
        rtf = total_time / audio_duration if audio_duration > 0 else 0
        print(
            f"[TTS] Complete: {audio_duration:.2f}s audio in {total_time:.2f}s (RTF={rtf:.2f})")

    except Exception as e:
        print(f"[TTS] Error: {e}")
        chunk_queue.put(e)
    finally:
        chunk_queue.put(None)  # Signal completion


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    global model
    print("Loading VoxCPM model...")
    start = time.time()
    model = VoxCPM.from_pretrained(
        "openbmb/VoxCPM0.5",
        load_denoiser=True,  # Skip denoiser for faster startup
        optimize=False,  # Disable torch.compile to avoid CUDA graph threading issues
    )
    print(f"Model loaded in {time.time() - start:.2f}s")
    print(f"Sample rate: {model.tts_model.sample_rate} Hz")
    yield
    print("Shutting down...")
    executor.shutdown(wait=False)


app = FastAPI(
    title="VoxCPM TTS API",
    description="OpenAI-compatible TTS streaming API powered by VoxCPM",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
async def serve_frontend():
    """Serve the HTML frontend."""
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.post("/v1/audio/speech")
async def create_speech(request: TTSRequest):
    """
    OpenAI-compatible TTS endpoint with streaming support.

    Streams audio chunks as they are generated for low latency.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not request.input.strip():
        raise HTTPException(status_code=400, detail="Input text is required")

    # Map speed to inference_timesteps (higher speed = fewer steps)
    # speed 1.0 -> 4 steps (fast), speed 0.5 -> 8 steps (quality)
    # Using fewer steps by default for real-time performance (RTF < 1)
    inference_timesteps = max(4, min(10, int(4 / request.speed)))

    sample_rate = model.tts_model.sample_rate
    is_wav = request.response_format != "pcm"

    # Determine content type
    content_type = "audio/wav" if is_wav else "audio/pcm"

    # Create a queue for streaming chunks between threads
    chunk_queue = queue.Queue()

    # Start generation in background thread
    executor.submit(
        generate_tts_chunks,
        request.input,
        inference_timesteps,
        sample_rate,
        is_wav,
        chunk_queue,
    )

    async def async_generator():
        """Async generator that reads from the queue."""
        loop = asyncio.get_event_loop()
        while True:
            # Non-blocking queue get with timeout
            try:
                item = await loop.run_in_executor(None, lambda: chunk_queue.get(timeout=60))
            except queue.Empty:
                print("[TTS] Timeout waiting for chunk")
                break

            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            yield item

    return StreamingResponse(
        async_generator(),
        media_type=content_type,
        headers={
            "X-Sample-Rate": str(sample_rate),
            "Transfer-Encoding": "chunked",
        }
    )


@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI-compatible)."""
    return {
        "object": "list",
        "data": [
            {
                "id": "tts-1",
                "object": "model",
                "created": 1699000000,
                "owned_by": "voxcpm",
            },
            {
                "id": "voxcpm-1.5",
                "object": "model",
                "created": 1699000000,
                "owned_by": "voxcpm",
            }
        ]
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "sample_rate": model.tts_model.sample_rate if model else None,
    }


# Mount static files (if directory exists)
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
