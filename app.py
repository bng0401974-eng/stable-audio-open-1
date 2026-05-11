import gradio as gr
import torch
import spaces
from diffusers import StableAudioPipeline
import scipy.io.wavfile as wavfile
import numpy as np

# Вчитување на моделот на CPU (ZeroGPU ќе го префрли на CUDA кога ќе треба)
model_id = "stabilityai/stable-audio-open-1.0"
pipe = StableAudioPipeline.from_pretrained(model_id, torch_dtype=torch.float16)

@spaces.GPU(duration=60)
def generate_audio(prompt, duration=5):
    # Се активира само за време на генерирање
    pipe.to("cuda")
    
    audio = pipe(
        prompt,
        num_inference_steps=200,
        audio_end_in_s=float(duration)
    ).audios[0]
    
    # Конверзија во WAV формат (44.1kHz)
    audio_np = audio.T.cpu().numpy()
    audio_int16 = (audio_np * 32767).astype(np.int16)
    
    output_path = "generated_sound.wav"
    wavfile.write(output_path, 44100, audio_int16)
    
    return output_path

# UI за Audio Creator-от
with gr.Blocks(title="LATIVM Audio Engine") as demo:
    gr.Markdown("# 🎵 LATIVM AI Audio Engine")
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(label="Опис на звукот", placeholder="пр. Crystal clear bell sound for children's app...")
            sec_slider = gr.Slider(minimum=1, maximum=30, value=5, label="Траење (секунди)")
            run_btn = gr.Button("Генерирај", variant="primary")
        with gr.Column():
            output_audio = gr.Audio(label="Резултат (WAV)")

    run_btn.click(fn=generate_audio, inputs=[input_text, sec_slider], outputs=output_audio)

demo.launch()