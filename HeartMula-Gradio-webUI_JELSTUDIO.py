import gradio as gr
import torch
import tempfile
import os
import datetime
import hashlib
import random
import numpy as np

from heartlib import HeartMuLaGenPipeline


PIPELINE = None


def random_seed():
    import random
    return random.randint(0, 2**32 - 1)

def load_pipeline(
    model_path,
    version,
    mula_device,
    codec_device,
    mula_dtype,
    codec_dtype,
    lazy_load,
):
    global PIPELINE
    if PIPELINE is None:
        PIPELINE = HeartMuLaGenPipeline.from_pretrained(
            model_path,
            device={
                "mula": torch.device(mula_device),
                "codec": torch.device(codec_device),
            },
            dtype={
                "mula": getattr(torch, mula_dtype),
                "codec": getattr(torch, codec_dtype),
            },
            version=version,
            lazy_load=lazy_load,
        )
    return PIPELINE


def load_text_input(value):
    if os.path.isfile(value):
        with open(value, "r", encoding="utf-8") as f:
            return f.read()
    return value


def generate_music(
    model_path,
    version,
    lyrics_text,
    tags_text,
    max_audio_length_ms,
    topk,
    temperature,
    cfg_scale,
    mula_device,
    codec_device,
    mula_dtype,
    codec_dtype,
    lazy_load,
    seed,
    audio_format,
):
    # Seed
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    pipe = load_pipeline(
        model_path,
        version,
        mula_device,
        codec_device,
        mula_dtype,
        codec_dtype,
        lazy_load,
    )

    lyrics = load_text_input(lyrics_text)
    tags = load_text_input(tags_text)

    os.makedirs("outputs", exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    content_hash = hashlib.sha1((lyrics + tags).encode("utf-8")).hexdigest()[:6]

    # Choose extension based on user selection
    ext = audio_format.lower()
    filename = f"heartmula_{timestamp}_{content_hash}.{ext}"
    output_path = os.path.join("outputs", filename)

    with torch.no_grad():
        pipe(
            {
                "lyrics": lyrics,
                "tags": tags,
            },
            max_audio_length_ms=max_audio_length_ms,
            save_path=output_path,
            topk=topk,
            temperature=temperature,
            cfg_scale=cfg_scale,
        )

    return output_path


with gr.Blocks(title="HeartMuLaGen Music Generator") as demo:
    gr.Markdown("## 🎵 HeartMuLaGen Music Generator")

    with gr.Row():
        model_path = gr.Textbox(
            label="Model Path",
            value="./ckpt",   # <-- Default path
        )
        version = gr.Textbox(label="Model Version", value="3B")

    lyrics = gr.Textbox(
        label="Lyrics (text or path to .txt file)",
        lines=8,
    )

    tags = gr.Textbox(
        label="Tags / Prompt (text or path to .txt file)",
        lines=4,
    )

    with gr.Row():
        max_audio_length_ms = gr.Slider(
            10_000, 600_000, value=240_000, step=1_000,
            label="Max Audio Length (ms)"
        )
        topk = gr.Slider(1, 200, value=50, step=1, label="Top-k")
        temperature = gr.Slider(0.1, 2.0, value=1.0, step=0.05, label="Temperature")
        cfg_scale = gr.Slider(0.1, 5.0, value=1.5, step=0.1, label="CFG Scale")

    gr.Markdown("### ⚙️ Advanced Settings")

    with gr.Row():
        mula_device = gr.Textbox(label="MULA Device", value="cuda")
        codec_device = gr.Textbox(label="Codec Device", value="cuda")

    with gr.Row():
        mula_dtype = gr.Dropdown(
            ["float32", "float16", "bfloat16"],
            value="bfloat16",
            label="MULA Dtype",
        )
        codec_dtype = gr.Dropdown(
            ["float32", "float16", "bfloat16"],
            value="float32",
            label="Codec Dtype",
        )

    lazy_load = gr.Checkbox(label="Lazy Load", value=True)
    #seed = gr.Number(label="Seed", value=248, precision=0)
    with gr.Row():
        seed = gr.Number(label="Seed", value=248, precision=0)
        random_seed_btn = gr.Button("🎲 Random Seed")

    # NEW: Audio format selector
    audio_format = gr.Dropdown(
        ["wav", "mp3", "flac"],
        value="flac",   # <-- Default to FLAC
        label="Output Audio Format",
    )

    generate_btn = gr.Button("🎶 Generate Music")

    output_audio = gr.Audio(label="Generated Audio", type="filepath")

    generate_btn.click(
        fn=generate_music,
        inputs=[
            model_path,
            version,
            lyrics,
            tags,
            max_audio_length_ms,
            topk,
            temperature,
            cfg_scale,
            mula_device,
            codec_device,
            mula_dtype,
            codec_dtype,
            lazy_load,
            seed,
            audio_format,
        ],
        outputs=output_audio,
    )

    random_seed_btn.click(
        fn=random_seed,
        inputs=[],
        outputs=seed,
    )


if __name__ == "__main__":
    demo.launch()