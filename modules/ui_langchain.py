import gradio as gr
import traceback

from modules import loaders, presets, shared, ui, utils
from modules.utils import gradio

from modules.models_settings import (
    apply_model_settings_to_state,
    save_model_settings,
    update_model_parameters
)
from modules.text_generation import (
    generate_reply_wrapper,
    generate_reply,
    stop_everything_event
)
from modules.prompts import count_tokens, load_prompt
from modules.langchains import (
    extract_template_input_variables,
    promptTemplate_to_prompt,
    llm_generate,
    )

inputs = ('langchain_template', 'interface_state',"langchain_slot")
outputs = ('langchain_output')

def create_ui(default_preset):
    langchain_params = presets.load_preset(default_preset)
    
    with gr.Tab("Langchain🦜️🔗", elem_id="Langchain-tab"):
        shared.gradio['last_input-langchain'] = gr.State('')
        with gr.Row():
            shared.gradio['langchain_loader'] = gr.Dropdown(label="langchain loader", choices=["Transformers", "OpenAI"], value=None)
            shared.gradio['langchain_embeding'] = gr.Dropdown(choices=utils.get_available_models(), value=default_preset, label='langchain_embeding', elem_classes='slim-dropdown')

        with gr.Row():
            with gr.Column():
                with gr.Box():
                    with gr.Column():
                        shared.gradio['langchain_temperature'] = gr.Slider(0.1, 1.99, value=langchain_params['temperature'], step=0.01, label='langchain_temperature',interactive=False)
                        shared.gradio['langchain_top_p'] = gr.Slider(0.0, 1.0, value=langchain_params['top_p'], step=0.01, label='langchain_top_p',interactive=False)
                        shared.gradio['langchain_verbose'] = gr.Checkbox(label="verbose", value=shared.args.langchain_verbose)
            with gr.Column():
                with gr.Accordion("Learn more", open=False):
                    gr.Markdown("""
                                ### specification
                                the langchain_temperature and langchain_top_p are equal to Parameters tab's temperature and top_p.
                                
                                """, elem_classes="markdown")
        with gr.Row():
            with gr.Column():
                shared.gradio['langchain_template'] = gr.Textbox(value='', elem_classes=['textbox_default', 'add_scrollbar'], lines=5, label='question_template')
                shared.gradio['langchain_input_variables'] = gr.Textbox(value='', elem_classes=['textbox_default', 'add_scrollbar'], label='input_variables',interactive=False)
                with gr.Row():
                    shared.gradio['langchain_slot'] = gr.Textbox(value='', elem_classes=['textbox_default', 'add_scrollbar'], lines=5, label='slot')
                shared.gradio['langchain_final_input'] = gr.Textbox(value='', elem_classes=['textbox_default', 'add_scrollbar'], label='final_input',interactive=False)
                shared.gradio['token-counter-langchain'] = gr.HTML(value="<span>0</span>", elem_classes=["token-counter", "default-token-counter"])
                with gr.Row():
                    shared.gradio['Generate-prompt'] = gr.Button('Generate_prompt', variant='primary',)
                    shared.gradio['Generate-langchain'] = gr.Button('Generate')
                    shared.gradio['Stop-langchain'] = gr.Button('Stop', elem_id='stop')
                    shared.gradio['Continue-langchain'] = gr.Button('Continue')
                with gr.Row():
                    shared.gradio['prompt_menu-langchain'] = gr.Dropdown(choices=utils.get_available_prompts(), value='None', label='Prompt', elem_classes='slim-dropdown')
                    ui.create_refresh_button(shared.gradio['prompt_menu-langchain'], lambda: None, lambda: {'choices': utils.get_available_prompts()}, 'refresh-button')
                    shared.gradio['save_prompt-langchain'] = gr.Button('💾', elem_classes='refresh-button')
                    shared.gradio['delete_prompt-langchain'] = gr.Button('🗑️', elem_classes='refresh-button')

            with gr.Column():
                shared.gradio['langchain_output'] = gr.Textbox(elem_classes=['textbox_default_output', 'add_scrollbar'],lines=20, label='Output')

        with gr.Accordion("Learn more", open=False):
            gr.Markdown("""### langchain""", elem_classes="markdown")
            
        with gr.Row():
                shared.gradio['langchains_status'] = gr.Markdown('No traceback',elem_classes="markdown")

def status():
    exc = traceback.format_exc()
    # print(exc)
    yield exc.replace('\n', '\n\n')


def create_event_handlers():

    shared.gradio['langchain_loader'].change(loaders.make_loader_params_visible, gradio('langchain_loader'), gradio(loaders.get_all_params()))
    shared.gradio['langchain_embeding'].change(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        apply_model_settings_to_state, gradio('langchain_embeding', 'interface_state'), gradio('interface_state')).then(
        ui.apply_interface_values, gradio('interface_state'), gradio(ui.list_interface_input_elements()), show_progress=False).then(
        update_model_parameters, gradio('interface_state'), None)
    
    shared.gradio['Generate-prompt'].click(
        extract_template_input_variables,gradio("langchain_template"),gradio('langchain_input_variables')).then(
        promptTemplate_to_prompt,gradio("langchain_template","langchain_slot"),gradio("langchain_final_input","langchains_status"),show_progress=False).then(
        status, None, gradio('langchains_status')
        )


    shared.gradio['Generate-langchain'].click(
        lambda x: x, gradio('langchain_final_input'), gradio('last_input-langchain')).then(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        update_model_parameters, gradio('interface_state'), None).then(
        llm_generate, gradio(inputs), gradio(outputs), show_progress=False).then(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        lambda: None, None, None, _js=f'() => {{{ui.audio_notification_js}}}').then(
        status, None, gradio('langchains_status')
        )
    
    shared.gradio['langchain_final_input'].submit(
        lambda x: x, gradio('langchain_final_input'), gradio('last_input-langchain')).then(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        generate_reply, gradio(inputs), gradio(outputs), show_progress=False).then(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        lambda: None, None, None, _js=f'() => {{{ui.audio_notification_js}}}')

    shared.gradio['Continue-langchain'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        generate_reply, [shared.gradio['langchain_output']] + gradio(inputs)[1:], gradio(outputs), show_progress=False).then(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        lambda: None, None, None, _js=f'() => {{{ui.audio_notification_js}}}')

    shared.gradio['Stop-langchain'].click(stop_everything_event, None, None, queue=False)

    shared.gradio['prompt_menu-langchain'].change(load_prompt, gradio('prompt_menu-langchain'), gradio('langchain_template'), show_progress=False)

    shared.gradio['save_prompt-langchain'].click(
        lambda x: x, gradio('langchain_slot'), gradio('save_contents')).then(
        lambda: 'prompts/', None, gradio('save_root')).then(
        lambda: utils.current_time() + '.txt', None, gradio('save_filename')).then(
        lambda: gr.update(visible=True), None, gradio('file_saver'))

    shared.gradio['delete_prompt-langchain'].click(
        lambda: 'prompts/', None, gradio('delete_root')).then(
        lambda x: x + '.txt', gradio('prompt_menu-langchain'), gradio('delete_filename')).then(
        lambda: gr.update(visible=True), None, gradio('file_deleter'))

    shared.gradio['langchain_final_input'].change(lambda x : f"<span>{count_tokens(x)}</span>", gradio('langchain_final_input'), gradio('token-counter-langchain'), show_progress=False)
