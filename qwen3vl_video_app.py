"""Qwen3-VL video understanding Gradio application."""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Optional

import imageio.v3 as iio
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
import gradio as gr

DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-30B-A3B-Instruct"
FALLBACK_MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"
FRAME_SAMPLE_LIMIT = 8


@dataclass
class LoadedModel:
    model_id: str
    processor: AutoProcessor
    model: AutoModelForVision2Seq


def detect_vram_gb() -> Optional[float]:
    if not torch.cuda.is_available():
        return None
    with contextlib.suppress(RuntimeError):
        device_properties = torch.cuda.get_device_properties(0)
        return float(device_properties.total_memory) / 1e9
    return None


def pick_model_id() -> str:
    vram_gb = detect_vram_gb()
    if vram_gb is None:
        print("⚠️ CUDA not available – defaulting to CPU execution. Consider enabling a GPU for best results.")
        return FALLBACK_MODEL_ID

    print(f"🖥️ Detected GPU VRAM: {vram_gb:.1f} GB")
    if vram_gb < 40:
        print(
            "💡 Automatically switching to the 7B vision-language model because the detected GPU has less than 40 GB VRAM."
        )
        return FALLBACK_MODEL_ID

    return DEFAULT_MODEL_ID


def load_model() -> LoadedModel:
    model_id = pick_model_id()
    print(f"🔄 Loading processor and model from {model_id}...")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    model_kwargs = {"trust_remote_code": True}
    if torch.cuda.is_available():
        model_kwargs.update({"device_map": "auto", "dtype": torch.bfloat16})
    else:
        model_kwargs.update({"torch_dtype": torch.float32})

    model = AutoModelForVision2Seq.from_pretrained(model_id, **model_kwargs)

    print(f"✅ Loaded model type: {model.config.model_type}")
    print(f"🧠 Model class: {model.__class__.__name__}")
    return LoadedModel(model_id=model_id, processor=processor, model=model)


LOADED = load_model()


def describe_video(video_file: Optional[str], question: str = "Describe what is happening in this video.") -> str:
    if video_file is None:
        return "⚠️ Please upload a video."

    print(f"🎞️ Processing {video_file} ...")
    try:
        video_frames = iio.imread(video_file)[:FRAME_SAMPLE_LIMIT]
        inputs = LOADED.processor(text=question, videos=video_frames, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = inputs.to(LOADED.model.device)
        output_ids = LOADED.model.generate(**inputs, max_new_tokens=256)
        response = LOADED.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        return response.strip()
    except Exception as exc:  # pragma: no cover - surfaced to UI
        return f"❌ Error: {exc}"


iface = gr.Interface(
    fn=describe_video,
    inputs=[
        gr.Video(label="🎥 Upload a video"),
        gr.Textbox(label="🧠 Question / Prompt", value="Describe what is happening in this video."),
    ],
    outputs=gr.Textbox(label="🗣️ Model Response"),
    title="Qwen3-VL Video Analyzer",
    description=(
        "Upload a video and ask a question — powered by Qwen3-VL with automatic model selection based on available VRAM."
    ),
)


iface.launch(server_name="0.0.0.0", server_port=7860)
