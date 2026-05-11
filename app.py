import os
import sys

# --- КРИТИЧЕН ХАК ЗА GRADIO BUG (TypeError: argument of type 'bool' is not iterable) ---
# Ова мора да биде на самиот почеток, пред било кој друг импорт од gradio
from gradio import routes


def mock_api_info(self): return {}


routes.App.api_info = mock_api_info
# --------------------------------------------------------------------------------------

import gradio as gr
import torch
from diffusers import StableAudioPipeline
import scipy.io.wavfile

# Глобална променлива за моделот
pipe = None


def load_model():
    global pipe
    if pipe is None:
        print("Вчитувам модел (ова може да потрае)...")
        # Користиме CPU верзија за стабилност на Hugging Face Free Tier
        pipe = StableAudioPipeline.from_pretrained(
            "facebook/stable-audio-open-1.0",
            torch_dtype=torch.float32
        )
        pipe = pipe.to("cpu")
    return pipe


def generate_audio(prompt, seconds, mode_selection):
    if not prompt:
        return None

    # Автоматско оптимизирање на промптот за LATIVM Loop системот
    final_prompt = prompt
    if mode_selection == "Loop":
        final_prompt = f"{prompt}, seamless loop, repetitive beat, consistent rhythm, high quality"
    else:
        final_prompt = f"{prompt}, high quality, musical composition"

    try:
        model = load_model()

        # Генерирање
        output = model(
            final_prompt,
            audio_end_in_s=seconds,
            num_inference_steps=50
        ).audios[0]

        # Зачувување во привремен фајл
        output_path = "lativm_output.wav"
        audio_data = output.t().numpy()
        scipy.io.wavfile.write(output_path, 44100, audio_data)

        return output_path
    except Exception as e:
        print(f"Грешка при генерирање: {e}")
        return None


# --- LATIVM UI ДИЗАЈН ---
custom_css = """
body { background-color: white !important; }
#title { color: #c305f7; text-align: center; font-family: 'Arial'; font-weight: bold; font-size: 2.8em; margin-bottom: 0px; }
.generate-btn { 
    background-color: #c305f7 !important; 
    border: none !important; 
    color: white !important; 
    font-weight: bold !important;
    height: 50px;
}
"""

with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎵 LATIVM AI Audio Engine", elem_id="title")
    gr.Markdown(
        "<p style='text-align: center; font-size: 1.2em; color: #666;'>AI Музички Генератор за Професионални Лопови</p>")

    with gr.Row():
        with gr.Column(scale=1):
            prompt_input = gr.Textbox(
                label="Опис на звукот",
                placeholder="На пр: Dark techno warehouse beat, 128 BPM, heavy industrial drums...",
                lines=4
            )

            with gr.Row():
                mode_radio = gr.Radio(
                    choices=["Loop", "Single Shot"],
                    value="Loop",
                    label="Режим"
                )
                duration_slider = gr.Slider(
                    minimum=5,
                    maximum=30,
                    step=1,
                    value=15,
                    label="Секунди"
                )

            generate_btn = gr.Button("ГЕНЕРИРАЈ", variant="primary", elem_classes="generate-btn")

        with gr.Column(scale=1):
            audio_output = gr.Audio(label="Генерирано аудио", type="filepath")
            gr.Markdown("""
            ### Инструкции за LATIVM:
            * **За подобар Loop:** Секогаш наведувај BPM (ритам).
            * **Квалитет:** Опиши го амбиентот (на пр. 'studio quality', 'reverb').
            * **Инструменти:** Наведи ги специфично (на пр. 'analog synth', '808 kick').
            """)

    generate_btn.click(
        fn=generate_audio,
        inputs=[prompt_input, duration_slider, mode_radio],
        outputs=audio_output
    )

# СТАРТУВАЊЕ (Критично за Docker на Hugging Face)
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_api=False  # Го спречува TypeError багот при подигнување
    )