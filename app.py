import os
import gradio as gr
import torch
from diffusers import StableAudioPipeline
import scipy.io.wavfile

# Моделот се вчитува само еднаш
pipe = None


def load_model():
    global pipe
    if pipe is None:
        pipe = StableAudioPipeline.from_pretrained("facebook/stable-audio-open-1.0", torch_dtype=torch.float32)
        pipe = pipe.to("cpu")
    return pipe


def generate_audio(prompt, seconds, mode_selection):
    if not prompt: return None

    final_prompt = prompt if mode_selection == "Single Shot" else f"{prompt}, seamless loop, rhythmic"

    try:
        model = load_model()
        output = model(final_prompt, audio_end_in_s=seconds, num_inference_steps=50).audios[0]

        output_path = "generated_output.wav"
        audio_data = output.t().numpy()
        scipy.io.wavfile.write(output_path, 44100, audio_data)
        return output_path
    except Exception as e:
        print(f"Error: {e}")
        return None


# Дизајн
custom_css = "#title { color: #c305f7; text-align: center; font-weight: bold; }"

with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("# 🎵 LATIVM AI Audio Engine", elem_id="title")
    with gr.Row():
        with gr.Column():
            inp = gr.Textbox(label="Prompt", placeholder="Deep techno bassline...")
            mode = gr.Radio(["Loop", "Single Shot"], value="Loop", label="Mode")
            sec = gr.Slider(5, 30, value=15, label="Seconds")
            btn = gr.Button("GENERATE", variant="primary")
        with gr.Column():
            out = gr.Audio(label="Result", type="filepath")

    btn.click(generate_audio, [inp, sec, mode], out)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_api=False  # ОВА Е КЛУЧНО: Исклучува генерирање на API документација
    )