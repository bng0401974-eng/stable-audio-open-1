import sys

# --- КРИТИЧНА ПОПРАВКА ЗА PYTHON 3.13 ---
# Мора да биде на самиот врв, пред било кој друг импорт!
try:
    import audioop
except ImportError:
    try:
        import audioop_lts as audioop

        sys.modules["audioop"] = audioop
        print("Успешно вчитан audioop-lts хак.")
    except ImportError:
        print("ГРЕШКА: audioop-lts не е инсталиран. Додај го во requirements.txt!")

# Сега можеме безбедно да ги вчитаме Gradio и Pydub
import gradio as gr
import numpy as np

# Твојот специфичен дизајн (LATIVM Style)
custom_css = """
.gradio-container { background-color: white !important; }
#title { color: #c305f7; text-align: center; font-family: 'Arial'; font-weight: bold; }
.primary-btn { 
    background-color: #c305f7 !important; 
    border: none !important; 
    color: white !important; 
}
"""


def lativm_test(text):
    return f"LATIVM Engine е подготвен! Внесен промпт: {text}"


with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("# 🎵 LATIVM AI Music Gen", elem_id="title")
    gr.Markdown("<p style='text-align: center;'>Системски статус: <b>ONLINE (Python 3.13 Fix)</b></p>")

    with gr.Row():
        with gr.Column():
            inp = gr.Textbox(label="Опис на звукот", placeholder="Тестирај го системот тука...")
            mode = gr.Radio(choices=["Loop", "Single Shot"], value="Loop", label="Режим")
            btn = gr.Button("ГЕНЕРИРАЈ", variant="primary", elem_classes="primary-btn")

        with gr.Column():
            out = gr.Textbox(label="Статус на излез")

    btn.click(fn=lativm_test, inputs=inp, outputs=out)

if __name__ == "__main__":
    demo.launch()