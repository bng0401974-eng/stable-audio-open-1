import gradio as gr


# Лажна функција која само симулира работа
def fake_generate(prompt, seconds):
    return None  # Не прави ништо, само за визуелен тест


# Твојот препознатлив стил: Бела позадина и пурпурни детали
custom_css = """
.gradio-container { background-color: white !important; }
button.primary { background-color: #c305f7 !important; border: none !important; }
#title { color: #c305f7; font-family: 'Arial'; text-align: center; }
"""

with gr.Blocks(css=custom_css, theme=gr.themes.Default()) as demo:
    gr.Markdown("# 🎵 LATIVM AI Audio Engine", elem_id="title")

    with gr.Row():
        with gr.Column():
            inp = gr.Textbox(label="Опис на звукот", placeholder="Внеси промпт...")
            sec = gr.Slider(5, 30, value=15, label="Секунди")
            btn = gr.Button("Генерирај Аудио", variant="primary", elem_classes="primary-btn")

        with gr.Column():
            out = gr.Audio(label="WAV Резултат")

# Стартување во "Reload" мод
if __name__ == "__main__":
    demo.launch(inbrowser=True)