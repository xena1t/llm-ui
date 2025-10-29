# ==============================================================
# Qwen3-VL-30B Video Understanding App
# Run on RunPod GPU (CUDA 11.8)
# ==============================================================
import torch, gradio as gr, imageio.v3 as iio
from transformers import AutoProcessor, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen3-VL-30B-Instruct"

print("🔄 Loading model and processor...")
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    torch_dtype=torch.bfloat16, 
    device_map="auto", 
    trust_remote_code=True
)
print("✅ Model loaded!")

# ---------------------------------------------------------------
# Inference function
# ---------------------------------------------------------------
def describe_video(video_file, question="Describe what is happening in this video."):
    if video_file is None:
        return "⚠️ Please upload a video."

    print(f"🎞️ Processing {video_file} ...")
    try:
        # Read and sample up to 8 frames for efficiency
        video_frames = iio.imread(video_file)[:8]
        inputs = processor(text=question, videos=video_frames, return_tensors="pt").to("cuda")

        output_ids = model.generate(**inputs, max_new_tokens=256)
        response = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        return response.strip()
    except Exception as e:
        return f"❌ Error: {e}"

# ---------------------------------------------------------------
# Gradio interface
# ---------------------------------------------------------------
iface = gr.Interface(
    fn=describe_video,
    inputs=[
        gr.Video(label="🎥 Upload a video"),
        gr.Textbox(label="🧠 Question / Prompt", value="Describe what is happening in this video.")
    ],
    outputs=gr.Textbox(label="🗣️ Model Response"),
    title="Qwen3-VL-30B Video Analyzer",
    description="Upload a video and ask a question — powered by Qwen3-VL-30B.",
)

iface.launch(server_name="0.0.0.0", server_port=7860)
