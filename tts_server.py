"""
OpenAI-Compatible TTS Streaming Server using VoxCPM.

Provides:
- POST /v1/audio/speech - OpenAI-compatible TTS endpoint with streaming
- GET / - Serves the HTML frontend
"""

import struct
import time
import threading
import queue
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from voxcpm import VoxCPM

# Enable TensorFloat32 for better GPU performance
torch.set_float32_matmul_precision('high')


class TTSWorker:
    """
    Dedicated TTS worker thread that owns the CUDA context.
    All TTS operations run in this single thread to keep CUDA graphs working.
    """

    def __init__(self):
        self.model: Optional[VoxCPM] = None
        self.request_queue = queue.Queue()
        self.thread: Optional[threading.Thread] = None
        self.running = False

    def start(self):
        """Start the worker thread and load the model."""
        self.running = True
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()

        # Wait for model to load
        ready_event = threading.Event()
        self.request_queue.put(("init", None, None, ready_event))
        ready_event.wait()
        print(
            f"TTS Worker ready. Sample rate: {self.model.tts_model.sample_rate} Hz")

    def stop(self):
        """Stop the worker thread."""
        self.running = False
        self.request_queue.put(("stop", None, None, None))
        if self.thread:
            self.thread.join(timeout=5)

    def _worker_loop(self):
        """Main worker loop - runs in dedicated thread with CUDA context."""
        while self.running:
            try:
                cmd, args, result_queue, event = self.request_queue.get(
                    timeout=1)
            except queue.Empty:
                continue

            if cmd == "stop":
                break
            elif cmd == "init":
                self._init_model()
                event.set()
            elif cmd == "generate":
                self._generate(args, result_queue)

    def _init_model(self):
        """Initialize model in the worker thread."""
        print("Loading VoxCPM model in TTS worker thread...")
        start = time.time()
        self.model = VoxCPM.from_pretrained(
            "openbmb/VoxCPM1.5",
            load_denoiser=False,
            optimize=True,  # Full CUDA optimizations
        )
        print(f"Model loaded in {time.time() - start:.2f}s")

    def _generate(self, args, result_queue):
        """Generate TTS in the worker thread."""
        text, inference_timesteps, is_wav = args
        sample_rate = self.model.tts_model.sample_rate
        generation_start = time.time()
        first_chunk = True
        total_samples = 0

        try:
            for chunk in self.model.generate_streaming(
                text=text,
                prompt_wav_path="/home/serj/tmp/example2.wav",
                prompt_text="All right, so have you ever heard of a little thing named text to speech? Well, it allows you to convert text into speech. I know that's super cool, isn't it?",
                cfg_value=2.5,
                inference_timesteps=30,
                normalize=False,
                denoise=False,
                retry_badcase=False,
            ):
                if first_chunk:
                    first_byte_time = time.time() - generation_start
                    print(f"[TTS] First chunk in {first_byte_time*1000:.0f}ms")

                    if is_wav:
                        result_queue.put(("header", sample_rate))

                    first_chunk = False

                # Convert to int16 bytes
                audio = np.clip(chunk, -1.0, 1.0)
                audio_int16 = (audio * 32767).astype(np.int16)
                total_samples += len(chunk)
                result_queue.put(("audio", audio_int16.tobytes()))

            total_time = time.time() - generation_start
            audio_duration = total_samples / sample_rate
            rtf = total_time / audio_duration if audio_duration > 0 else 0
            print(
                f"[TTS] Complete: {audio_duration:.2f}s audio in {total_time:.2f}s (RTF={rtf:.2f})")

        except Exception as e:
            print(f"[TTS] Error: {e}")
            result_queue.put(("error", str(e)))
        finally:
            result_queue.put(("done", None))

    def generate_streaming(self, text: str, inference_timesteps: int, is_wav: bool):
        """Submit a generation request and yield results."""
        result_queue = queue.Queue()
        self.request_queue.put(
            ("generate", (text, inference_timesteps, is_wav), result_queue, None))

        sample_rate = self.model.tts_model.sample_rate

        while True:
            msg_type, data = result_queue.get()

            if msg_type == "done":
                break
            elif msg_type == "error":
                raise RuntimeError(data)
            elif msg_type == "header":
                yield create_wav_header(data)
            elif msg_type == "audio":
                yield data

    @property
    def sample_rate(self):
        return self.model.tts_model.sample_rate if self.model else None


# Global worker
tts_worker = TTSWorker()


class TTSRequest(BaseModel):
    """OpenAI-compatible TTS request."""
    model: str = Field(
        default="tts-1", description="Model ID (ignored, uses VoxCPM)")
    input: str = Field(..., description="Text to synthesize")
    voice: str = Field(default="default", description="Voice name")
    response_format: str = Field(
        default="wav", description="Audio format: wav or pcm")
    speed: float = Field(default=1.0, ge=0.25, le=4.0,
                         description="Speed factor")


def create_wav_header(sample_rate: int, num_channels: int = 1, bits_per_sample: int = 16, data_size: int = 0) -> bytes:
    """Create a WAV file header."""
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8

    if data_size == 0:
        data_size = 0xFFFFFFFF - 36

    return struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF', data_size + 36, b'WAVE', b'fmt ', 16, 1,
        num_channels, sample_rate, byte_rate, block_align, bits_per_sample,
        b'data', data_size
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start TTS worker on startup."""
    tts_worker.start()
    yield
    tts_worker.stop()
    print("Shutting down...")


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
def create_speech(request: TTSRequest):
    """
    OpenAI-compatible TTS endpoint with streaming support.
    """
    if tts_worker.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not request.input.strip():
        raise HTTPException(status_code=400, detail="Input text is required")

    inference_timesteps = max(4, min(10, int(4 / request.speed)))
    is_wav = request.response_format != "pcm"
    content_type = "audio/wav" if is_wav else "audio/pcm"

    return StreamingResponse(
        tts_worker.generate_streaming(
            request.input, inference_timesteps, is_wav),
        media_type=content_type,
        headers={"X-Sample-Rate": str(tts_worker.sample_rate)},
    )


@app.get("/v1/models")
async def list_models():
    """List available models."""
    return {
        "object": "list",
        "data": [
            {"id": "tts-1", "object": "model",
                "created": 1699000000, "owned_by": "voxcpm"},
            {"id": "voxcpm-1.5", "object": "model",
                "created": 1699000000, "owned_by": "voxcpm"},
        ]
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": tts_worker.model is not None,
        "sample_rate": tts_worker.sample_rate,
    }


static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
