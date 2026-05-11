import os
import sys

# --- ХАК ЗА PYTHON 3.13 (audioop) ---
try:
    import audioop
except ImportError:
    try:
        import audioop_lts as audioop

        sys.modules["audioop"] = audioop
    except ImportError:
        pass

import gradio as gr
import torch
import scipy.io.wavfile
from diffusers import StableAudioPipeline

# --- ГЛОБАЛНИ ПРОМЕНЛИВИ ---
# Го оставаме pipe на None за да не троши RAM при самиот старт на билдот
pipe = None


def load_model():
    global pipe
    if pipe is None:
        print("Вчитување на LATIVM Engine (Stable Audio)...")
        # Користиме float32 бидејќи сме на CPU
        pipe = StableAudioPipeline.from_pretrained(
            "stabilityai/stable-audio-open-1.0",
            torch_dtype=torch.float32
        )
        pipe.to("cpu")
    return pipe


def generate_audio(prompt, seconds):
    if not prompt:
        return None

    try:
        model = load_model()

        # Оптимизирани параметри за CPU (за да не трае вечно)
        # 15-20 чекори е доволно за солиден звук на овој модел
        num_steps = 20

        print(f"Генерирање: {prompt} ({seconds} сек.)")

        output = model(
            prompt,
            num_inference_steps=num_steps,
            audio_end_in_s=seconds
        )

        audio_data = output.audios[0][0].numpy()
        sampling_rate = model.vae.sampling_rate

        output_path = "lativm_output.wav"
        scipy.io.wavfile.write(output_path, sampling_rate, audio_data)

        return output_path
    except Exception as e:
        print(f"Грешка при генерирање: {e}")
        return None


# --- ТВОЈОТ ПРЕПОЗНАТЛИВ UI (LATIVM СТИЛ) ---
custom_css = """
.gradio-container { background-color: white !important; }
#title { color: #c305f7; text-align: center; font-family: 'Arial', sans-serif; font-weight: bold; }
.primary-btn { 
    background-color: #c305f7 !important; 
    border: none !important; 
    color: white !important; 
    font-weight: bold !important;
}
.primary-btn:hover { background-color: #a104c9 !important; }
.footer { text-align: center; margin-top: 50px; color: #888; font-size: 0.8em; }
"""

with gr.Blocks(css=custom_css, theme=gr.themes.Default()) as demo:
    gr.Markdown("# 🎵 LATIVM AI Audio Engine", elem_id="title")

    with gr.Row():
        with gr.Column():
            prompt_input = gr.Textbox(
                label="Опис на звукот (Prompt)",
                placeholder="Внеси опис, на пр: Cinematic drums, deep bass, 120bpm...",
                lines=4
            )
            duration_slider = gr.Slider(
                minimum=5,
                maximum=30,
                value=15,
                step=1,
                label="Времетраење во секунди"
            )
            generate_btn = gr.Button("ГЕНЕРИРАЈ АУДИО", variant="primary", elem_classes="primary-btn")

        with gr.Column():
            audio_output = gr.Audio(
                label="Резултат (WAV)",
                type="filepath",
                interactive=False
            )

    gr.Markdown(
        "<div class='footer'>LATIVM Project © 2026<br>Базирано на Stable Audio Open 1.0</div>"
    )

    # Функционалност
    generate_btn.click(
        fn=generate_audio,
        inputs=[prompt_input, duration_slider],
        outputs=audio_output
    )

if __name__ == "__main__":
    # Hugging Face автоматски ја доделува портата
    demo.launch()