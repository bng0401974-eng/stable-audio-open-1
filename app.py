import sys
import os

# 1. Хак за Python 3.13 компатибилност со pydub
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

# 2. Вчитување на моделот (Stable Audio Open 1.0)
# Користиме float32 за CPU бидејќи float16 често не е поддржан на сите CPU сервери
print("Вчитување на LATIVM Engine...")
pipe = StableAudioPipeline.from_pretrained(
    "stabilityai/stable-audio-open-1.0",
    torch_dtype=torch.float32
)
pipe.to("cpu")


def generate_audio(prompt, seconds):
    if not prompt:
        return None

    # Пресметка на број на чекори (steps)
    # Помалку чекори = побрзо генерирање на бесплатен сервер
    num_steps = 20

    # Генерирање
    audio = pipe(
        prompt,
        num_inference_steps=num_steps,
        audio_end_in_s=seconds
    ).audios[0][0].numpy()

    # Зачувување во привремен фајл
    output_path = "generated_lativm.wav"
    sampling_rate = pipe.vae.sampling_rate
    scipy.io.wavfile.write(output_path, sampling_rate, audio)

    return output_path


# 3. Твојот препознатлив стил (UI)
custom_css = """
.gradio-container { background-color: white !important; }
button.primary { 
    background-color: #c305f7 !important; 
    border: none !important; 
    color: white !important;
}
button.primary:hover { background-color: #a104c9 !important; }
#title { color: #c305f7; font-family: 'Arial'; text-align: center; font-weight: bold; }
.description { text-align: center; color: #666; margin-bottom: 20px; }
"""

with gr.Blocks(css=custom_css, theme=gr.themes.Default()) as demo:
    gr.Markdown("# 🎵 LATIVM AI Audio Engine", elem_id="title")
    gr.Markdown("Креирај уникатни аудио примероци и лопови со помош на вештачка интелигенција.",
                elem_classes="description")

    with gr.Row():
        with gr.Column(scale=1):
            inp = gr.Textbox(
                label="Опис на звукот",
                placeholder="На пр. Techno bassline, 128 BPM, high quality...",
                lines=3
            )
            sec = gr.Slider(
                minimum=5,
                maximum=30,
                value=15,
                step=1,
                label="Времетраење (секунди)"
            )
            btn = gr.Button("ГЕНЕРИРАЈ АУДИО", variant="primary")

        with gr.Column(scale=1):
            out = gr.Audio(
                label="LATIVM WAV Резултат",
                type="filepath",
                interactive=False
            )

    # Поврзување на функцијата
    btn.click(
        fn=generate_audio,
        inputs=[inp, sec],
        outputs=out,
        api_name="generate"
    )

    gr.HTML("<p style='text-align: center; margin-top: 20px;'>© 2026 LATIVM Project - Powered by Stable Audio</p>")

if __name__ == "__main__":
    demo.launch()