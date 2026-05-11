import gradio as gr

def test_func(text):
    return f"LATIVM Online: {text}"

with gr.Blocks() as demo:
    gr.Markdown("# LATIVM System Check")
    inp = gr.Textbox(placeholder="Системот се подигнува...")
    out = gr.Textbox()
    btn = gr.Button("Тест")
    btn.click(test_func, inp, out)

demo.launch()