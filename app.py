import streamlit as st
import torch
from diffusers import StableDiffusionPipeline

if 'loaded_config' not in st.session_state:
    st.session_state.loaded_config = {
        'run_on_gpu': True,
        'half_precision': False
    }

if 'has_gpu' not in st.session_state:
    st.session_state.has_gpu = torch.cuda.is_available()

if 'pipe' not in st.session_state:
    with st.spinner('Loading pipeline ...'):
        st.session_state.pipe = StableDiffusionPipeline.from_pretrained('./stable-diffusion-v1-5')
        if st.session_state.has_gpu:
            st.session_state.pipe.to('cuda')


def update_config():
    if st.session_state.half_precision != st.session_state.loaded_config['half_precision']:
        del st.session_state.pipe
        torch.cuda.empty_cache()
        if st.session_state.half_precision:
            with st.spinner('Loading half precision pipe'):
                st.session_state.pipe = StableDiffusionPipeline.from_pretrained('./stable-diffusion-v1-5', torch_dtype=torch.float16, revision='fp16')
                st.session_state.loaded_config['half_precision'] = True
        else:
            with st.spinner('Loading full precision pipe'):
                st.session_state.pipe = StableDiffusionPipeline.from_pretrained('./stable-diffusion-v1-5')
                st.session_state.loaded_config['half_precision'] = False
        st.session_state.loaded_config['run_on_gpu'] = False
    
    if st.session_state.run_on_gpu != st.session_state.loaded_config['run_on_gpu']:
        if st.session_state.run_on_gpu:
            with st.spinner('Updating pipe to run on GPU'):
                st.session_state.pipe.to('cuda')
                st.session_state.loaded_config['run_on_gpu'] = True
        else:
            with st.spinner('Updating pipe to run on CPU'):
                st.session_state.pipe.to('cpu')
                st.session_state.loaded_config['run_on_gpu'] = False


def run_prompt():
    update_config()
    if st.session_state.prompt:
        with st.spinner('Processing prompt'):
            image = st.session_state.pipe(st.session_state.prompt).images[0]
        st.image(image)
        
        

with st.sidebar:
    st.title('Options')
    st.checkbox('Run on GPU', value=True, disabled=not st.session_state.has_gpu, key='run_on_gpu', help='Running on GPU brings a result much faster. It would not be available if not supported.')
    st.checkbox('Runt at half precision', key='half_precision')
    
st.text_input('Propt:', key='prompt')
st.button('Surprise me!', on_click=run_prompt)