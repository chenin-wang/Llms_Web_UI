import gradio as gr

from modules import loaders, presets, shared, ui, utils
from modules.utils import gradio

from modules.models_settings import (
    apply_model_settings_to_state,
    save_model_settings,
    update_model_parameters
)


def create_ui(default_preset):
    langchain_params = presets.load_preset(default_preset)

    with gr.Tab("Langchain🦜️🔗", elem_id="Langchain-tab"):
        with gr.Row():
            shared.gradio['langchain_loader'] = gr.Dropdown(label="langchain loader", choices=["Transformers", "OpenAI"], value=None)
            shared.gradio['langchain_embeding'] = gr.Dropdown(choices=utils.get_available_models(), value=default_preset, label='langchain_embeding', elem_classes='slim-dropdown')

        with gr.Row():
            with gr.Column():
                with gr.Box():
                    with gr.Column():
                        shared.gradio['langchain_temperature'] = gr.Slider(0.1, 1.99, value=langchain_params['temperature'], step=0.01, label='langchain_temperature',interactive=False)
                        shared.gradio['langchain_top_p'] = gr.Slider(0.0, 1.0, value=langchain_params['top_p'], step=0.01, label='langchain_top_p',interactive=False)
            with gr.Column():
                with gr.Accordion("Learn more", open=False):
                    gr.Markdown("""
                                ### specification
                                the langchain_temperature and langchain_top_p are equal to Parameters tab's temperature and top_p.
                                
                                """, elem_classes="markdown")

        with gr.Accordion("Learn more", open=False):
            gr.Markdown("""### langchain""", elem_classes="markdown")
            
        with gr.Row():
                shared.gradio['embedings_status'] = gr.Markdown('No embeding is loaded' if shared.langchain_embed_name == 'None' else 'Ready')


def create_event_handlers():

    shared.gradio['langchain_loader'].change(loaders.make_loader_params_visible, gradio('langchain_loader'), gradio(loaders.get_all_params()))
    shared.gradio['langchain_embeding'].change(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        apply_model_settings_to_state, gradio('langchain_embeding', 'interface_state'), gradio('interface_state')).then(
        ui.apply_interface_values, gradio('interface_state'), gradio(ui.list_interface_input_elements()), show_progress=False).then(
        update_model_parameters, gradio('interface_state'), None)