import gradio as gr
import torch
import spaces
from diffusers import StableAudioPipeline
import scipy.io.wavfile as wavfile
import numpy as np

# Дефинирање на моделот
model_id = "stabilityai/stable-audio-open-1.0"

# Вчитување на моделот (на CPU додека не затреба GPU)
pipe = StableAudioPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16
)


@spaces.GPU(duration=60)
def generate_audio(prompt, seconds=10):
    # Префрлање на моделот на CUDA само за време на пресметката
    pipe.to("cuda")

    # Генерирање на аудиото
    output = pipe(
        prompt,
        num_inference_steps=200,
        audio_end_in_s=float(seconds)
    ).audios[0]

    # Форматирање за зачувување (44.1kHz)
    audio_np = output.T.cpu().numpy()
    audio_int16 = (audio_np * 32767).astype(np.int16)

    output_path = "generated_audio.wav"
    wavfile.write(output_path, 44100, audio_int16)

    return output_path


# Креирање на UI со Gradio
with gr.Blocks(title="LATIVM Audio Engine") as demo:
    gr.Markdown("# 🎵 LATIVM AI Audio Generator")
    gr.Markdown("Користи ZeroGPU за генерирање звук преку Stable Audio Open 1.0")

    with gr.Row():
        with gr.Column():
            text_prompt = gr.Textbox(label="Опис на звукот", placeholder="пр. Upbeat techno beat for a mobile game...")
            duration = gr.Slider(minimum=1, maximum=30, value=10, label="Времетраење (секунди)")
            btn = gr.Button("Генерирај звук", variant="primary")
        with gr.Column():
            audio_out = gr.Audio(label="Резултат")

    btn.click(fn=generate_audio, inputs=[text_prompt, duration], outputs=audio_out)

demo.launch()