import sys

# --- FIX ЗА PYTHON 3.13 (audioop) ---
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


def generate_audio(prompt, seconds, mode_selection):
    # Ова ќе го додадеме кога ќе проработи интерфејсот
    return None


# --- LATIVM UI ---
custom_css = """
.gradio-container { background-color: white !important; }
#title { color: #c305f7; text-align: center; font-weight: bold; }
.primary-btn { background-color: #c305f7 !important; color: white !important; }
"""

with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("# 🎵 LATIVM AI Music Gen", elem_id="title")

    with gr.Row():
        with gr.Column():
            inp = gr.Textbox(label="Опис на звукот", placeholder="Cinematic drums...")
            mode = gr.Radio(choices=["Loop", "Single Shot"], value="Loop", label="Режим")
            sec = gr.Slider(5, 30, value=15, label="Секунди")
            btn = gr.Button("ГЕНЕРИРАЈ", variant="primary", elem_classes="primary-btn")

        with gr.Column():
            out = gr.Audio(label="Резултат")

    btn.click(fn=generate_audio, inputs=[inp, sec, mode], outputs=out)

if __name__ == "__main__":
    demo.launch()