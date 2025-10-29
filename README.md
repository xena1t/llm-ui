# Qwen3-VL Video Understanding App

This repository contains a ready-to-run Gradio interface for video captioning and reasoning with Qwen3/Qwen2 vision-language models. Follow the steps below to reproduce the full setup on RunPod, Colab, or any CUDA-enabled workstation.

---

## 1. Create and activate an environment

```bash
conda create -n qwen3vl python=3.10 -y
conda activate qwen3vl
```

> You can also use `python -m venv qwen3vl` if Conda is not available.

---

## 2. Install core dependencies

The PyPI CUDA wheels for PyTorch live on a separate index. Install PyTorch first, then the remaining packages:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

The `requirements.txt` file installs the latest `transformers` direct from the Hugging Face GitHub repository to ensure Qwen3 support, along with `accelerate`, `hf_transfer`, `gradio`, `imageio`, and optional `bitsandbytes` for 4-bit loading.

---

## 3. Authenticate with Hugging Face

Qwen3 models are gated. Log in with your Hugging Face token before running the app:

```bash
huggingface-cli login
# or
hf auth login
```

Generate a token at <https://huggingface.co/settings/tokens> and paste it when prompted. You should see `Login successful`.

---

## 4. Launch the Gradio app

```bash
python qwen3vl_video_app.py
```

When startup completes you will see output similar to:

```
Running on local URL:  http://0.0.0.0:7860
Running on public URL: https://<session-id>-7860.proxy.runpod.io
```

Open the public URL provided by your environment to access the UI. Upload an MP4/MOV file and optionally change the question prompt to obtain model responses.

---

## 5. Automatic model selection & GPU requirements

The script tries to load **`Qwen/Qwen3-VL-30B-A3B-Instruct`** by default. If CUDA is unavailable or the detected GPU VRAM is below **40 GB**, it automatically falls back to **`Qwen/Qwen2-VL-7B-Instruct`** so the app can run on 16–24 GB cards.

| Model ID                          | Description                                   | Recommended VRAM |
| --------------------------------- | --------------------------------------------- | ---------------- |
| `Qwen/Qwen3-VL-30B-A3B-Instruct`  | Flagship 30B vision-language model            | ≥ 40 GB          |
| `Qwen/Qwen2-VL-7B-Instruct`       | Balanced multimodal model for smaller GPUs    | 16–24 GB         |
| `Qwen/Qwen2-VL-2B-Instruct`       | Extra-light fallback (manually switch in code)| ≤ 12 GB          |

To capture GPU details at startup the script prints detected VRAM and the chosen model ID.

---

## 6. Troubleshooting checklist

| Symptom                                                         | Likely Cause                                           | Resolution |
| --------------------------------------------------------------- | ------------------------------------------------------ | ---------- |
| `Permission denied (publickey)` when cloning or pulling         | SSH key mismatch                                       | Load the correct private key (e.g., `~/.ssh/id_ed25519`) and add it to RunPod. |
| `401 Unauthorized` / gated model download errors                | Hugging Face token missing or expired                  | Run `huggingface-cli login` again with a valid token. |
| `hf_transfer not found`                                         | RunPod images set `HF_HUB_ENABLE_HF_TRANSFER=1`        | Install via `pip install hf_transfer`. |
| `KeyError: 'qwen3_vl_moe'` or similar from Transformers         | Transformers version too old                           | Use the provided `requirements.txt` (installs latest Git main). |
| `Unrecognized configuration class Qwen3VLMoeConfig`             | Loaded with `AutoModelForCausalLM`                     | Use `AutoModelForVision2Seq` (already set in `qwen3vl_video_app.py`). |
| Deprecation warning for `torch_dtype`                           | API change in Transformers                             | This repo sets `dtype=torch.bfloat16` when loading the model. |
| GPU out-of-memory errors when loading the 30B model             | Insufficient VRAM                                      | Let the script fall back to the 7B model or edit `DEFAULT_MODEL_ID`. |

---

## 7. Optional enhancements

- Adjust `FRAME_SAMPLE_LIMIT` in `qwen3vl_video_app.py` to analyze more than eight frames if you have sufficient VRAM.
- Uncomment or adapt the fallback logic to target `Qwen/Qwen2-VL-2B-Instruct` for CPU-only scenarios.
- Install [ffmpeg](https://ffmpeg.org/) if you encounter issues reading certain video formats with `imageio`.

---

## 8. Repository contents

- `qwen3vl_video_app.py` – main Gradio interface script with automatic model selection.
- `requirements.txt` – reproducible dependency list (install PyTorch separately as shown above).
- `README.md` – this setup guide and troubleshooting reference.

Happy video reasoning! 🎥🤖
