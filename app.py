import os
import torch
import spaces
from diffusers import StableAudioPipeline
import scipy.io.wavfile as wavfile
import numpy as np
import gradio as gr

# Глобална променлива која ќе биде None на почеток
pipe = None


def load_model():
    global pipe
    if pipe is None:
        model_id = "stabilityai/stable-audio-open-1.0"
        hf_token = os.getenv("HF_TOKEN")
        print("--- Вчитувам модел во GPU меморија ---")
        pipe = StableAudioPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            token=hf_token,
            low_cpu_mem_usage=True  # Ова е клучно за да не пукне RAM-от
        )
    return pipe


@spaces.GPU(duration=120)  # Зголемено траење за првото вчитување
def generate_audio(prompt, seconds=10):
    model = load_model()
    model.to("cuda")

    print(f"--- Генерирам аудио за: {prompt} ---")
    output = model(
        prompt,
        num_inference_steps=200,
        audio_end_in_s=float(seconds)
    ).audios[0]

    audio_np = output.T.cpu().numpy()
    audio_int16 = (audio_np * 32767).astype(np.int16)

    output_path = "output.wav"
    wavfile.write(output_path, 44100, audio_int16)
    return output_path


# UI дел кој е многу лесен и нема да го убие серверот при старт
with gr.Blocks() as demo:
    gr.Markdown("# 🎵 LATIVM AI Audio Engine (Lazy Loading Mode)")
    with gr.Row():
        with gr.Column():
            inp = gr.Textbox(label="Опис на звукот")
            sec = gr.Slider(5, 30, value=10, label="Секунди")
            btn = gr.Button("Генерирај (Првото кликање ќе трае подолго)")
        with gr.Column():
            out = gr.Audio(label="WAV Резултат")

    btn.click(fn=generate_audio, inputs=[inp, sec], outputs=out)

demo.launch()