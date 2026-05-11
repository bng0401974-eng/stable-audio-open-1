import os
import torch
import spaces
from diffusers import StableAudioPipeline
import scipy.io.wavfile as wavfile
import numpy as np
import gradio as gr

# Глобална променлива за моделот
pipe = None


def load_model():
    global pipe
    if pipe is None:
        model_id = "stabilityai/stable-audio-open-1.0"
        hf_token = os.getenv("HF_TOKEN")

        print("--- Вчитувам модел (Lazy Loading) ---")

        # Го вчитуваме моделот директно во соодветниот формат
        # Не користиме .to("cuda") тука, spaces.GPU ќе го реши тоа подоцна
        pipe = StableAudioPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            token=hf_token,
            low_cpu_mem_usage=True
        )
    return pipe


@spaces.GPU(duration=120)
def generate_audio(prompt, seconds=10):
    # Повикување на моделот
    model = load_model()

    # КЛУЧНО: Кај ZeroGPU НЕ користиме model.to("cuda")
    # Декораторот @spaces.GPU автоматски го префрла моделот на графичка

    print(f"--- Започнувам генерирање: {prompt} ---")

    try:
        # Генерирање на аудиото
        output = model(
            prompt,
            num_inference_steps=200,
            audio_end_in_s=float(seconds)
        ).audios[0]

        # Конверзија во формат погоден за WAV фајл
        audio_np = output.T.cpu().numpy()
        audio_int16 = (audio_np * 32767).astype(np.int16)

        output_path = "output.wav"
        wavfile.write(output_path, 44100, audio_int16)

        print("--- Успешно генерирано аудио! ---")
        return output_path

    except Exception as e:
        print(f"Грешка при генерирање: {e}")
        return None


# Интерфејс (Gradio)
with gr.Blocks(theme=gr.themes.Default(primary_hue="purple")) as demo:
    gr.Markdown("# 🎵 LATIVM AI Audio Engine")
    gr.Markdown("Внесете опис и почекајте моделот да се активира на GPU.")

    with gr.Row():
        with gr.Column():
            inp = gr.Textbox(
                label="Промпт (на англиски)",
                placeholder="Пример: Lo-fi hip hop beat, calm, 120 BPM, melodic loop"
            )
            sec = gr.Slider(minimum=5, maximum=30, value=15, step=1, label="Должина во секунди")
            btn = gr.Button("Генерирај Аудио", variant="primary")

        with gr.Column():
            out = gr.Audio(label="Резултат", type="filepath")

    btn.click(
        fn=generate_audio,
        inputs=[inp, sec],
        outputs=out
    )

# Стартување
if __name__ == "__main__":
    demo.launch()