import os
import sys

# 1. Специфична поправка за аудио модулите (за секој случај)
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
from diffusers import StableAudioPipeline
import scipy.io.wavfile

# Глобална променлива за моделот за да не се вчитува постојано
pipe = None


def load_model():
    global pipe
    if pipe is None:
        # Користиме Stable Audio Open моделот
        pipe = StableAudioPipeline.from_pretrained("facebook/stable-audio-open-1.0", torch_dtype=torch.float32)
        pipe = pipe.to("cpu")  # Одиме на CPU за стабилност во почеток
    return pipe


def generate_audio(prompt, seconds, mode_selection):
    if not prompt:
        return None

    # Модификација на промптот за Loop опцијата
    final_prompt = prompt
    if mode_selection == "Loop":
        final_prompt = f"{prompt}, seamless loop, rhythmic, repetitive beat"

    try:
        model = load_model()

        # Генерирање на аудиото
        output = model(
            final_prompt,
            audio_end_in_s=seconds,
            prompt_audios=None,  # За почеток без аудио-референца
            num_inference_steps=50
        ).audios[0]

        # Зачувување во привремен фајл
        output_path = "generated_output.wav"
        # Конверзија за scipy (од 1, T во T, 1)
        audio_data = output.t().numpy()
        scipy.io.wavfile.write(output_path, 44100, audio_data)

        return output_path
    except Exception as e:
        print(f"Грешка при генерирање: {e}")
        return None


# --- LATIVM CUSTOM UI ---
custom_css = """
body { background-color: white !important; }
.gradio-container { background-color: white !important; }
#title { color: #c305f7; text-align: center; font-family: 'Arial'; font-weight: bold; font-size: 2.5em; margin-bottom: 10px; }
.primary-btn { 
    background-color: #c305f7 !important; 
    border: none !important; 
    color: white !important; 
    font-weight: bold !important;
}
.primary-btn:hover { background-color: #a104cc !important; }
"""

with gr.Blocks(css=custom_css, theme=gr.themes.Default()) as demo:
    gr.Markdown("# 🎵 LATIVM AI Audio Engine", elem_id="title")
    gr.Markdown("<p style='text-align: center;'>Професионално AI генерирање на музички лопови и звуци.</p>")

    with gr.Row():
        with gr.Column(scale=1):
            inp = gr.Textbox(
                label="Опис на звукот (Prompt)",
                placeholder="На пр: Lo-fi hip hop beat, 90bpm, chill vibes...",
                lines=3
            )

            mode = gr.Radio(
                choices=["Loop", "Single Shot"],
                value="Loop",
                label="Режим на работа"
            )

            sec = gr.Slider(minimum=5, maximum=30, step=1, value=15, label="Времетраење (секунди)")

            btn = gr.Button("ГЕНЕРИРАЈ АУДИО", variant="primary", elem_classes="primary-btn")

        with gr.Column(scale=1):
            out = gr.Audio(label="Резултат", type="filepath")
            gr.Markdown("### Како да добиеш подобар Loop?")
            gr.Markdown("* Опиши го ритамот (BPM)\n* Наведи ги инструментите\n* Избегнувај премногу комплексни вокали")

    btn.click(
        fn=generate_audio,
        inputs=[inp, sec, mode],
        outputs=out
    )

# КРИТИЧНО ЗА DOCKER: server_name мора да е 0.0.0.0 и порта 7860
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)