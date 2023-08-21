import base64
import copy
import functools
import json
import re
from pathlib import Path
import importlib

from typing import TYPE_CHECKING, Optional, Tuple, Union, Callable, List, Any, Generator
import gradio as gr
import yaml
from PIL import Image

import modules.shared as shared
from modules.extensions import apply_extensions
from modules.html_generator import chat_html_wrapper, make_thumbnail
from modules.logging_colors import logger
from modules.text_generation import (
    generate_reply,
    get_encoded_length,
    get_max_prompt_length
)
from modules.utils import (
    delete_file,
    get_available_characters,
    replace_all,
    save_file
)


from langchain.llms import HuggingFacePipeline, HuggingFaceHub
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from langchain import PromptTemplate, LLMChain
from langchain.embeddings import HuggingFaceEmbeddings,HuggingFaceHubEmbeddings

INIT_LLMS_DOCSTRING=r"""currently only ('text2text-generation', 'text-generation') are supported"""


def add_end_docstrings(*docstr):
    def docstring_decorator(fn):
        fn.__doc__ = (fn.__doc__ if fn.__doc__ is not None else "") + "".join(docstr)
        return fn

    return docstring_decorator


template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])


# https://colab.research.google.com/drive/1h2505J5H4Y9vngzPD08ppf1ga8sWxLvZ?usp=sharing#scrollTo=wEKTalGcgxRg

# currently only ('text2text-generation', 'text-generation') are supported

@add_end_docstrings(INIT_LLMS_DOCSTRING)
def llm_chain_generate(
    is_hub:Optional[bool(object)] = False,
):
    """
    generate the llm chain.
    """

    if is_hub:
        llm_chain = LLMChain(prompt=prompt, 
                            llm=HuggingFaceHub(repo_id="google/flan-t5-xl", 
                                            model_kwargs={"temperature":0, 
                                                            "max_length":64}))
    else:
        pipe = pipeline(
            "text-generation", model=shared.model, tokenizer=shared.tokenizer, max_new_tokens=10,
        )
        local_llm = HuggingFacePipeline(pipeline=pipe)

        llm_chain = LLMChain(prompt=prompt, 
                            llm=local_llm
                            )
    
    return llm_chain



# local
# model_id = "gpt2"
# tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = AutoModelForCausalLM.from_pretrained(model_id)

def llm_chain_run(llm_chain=None,question=""):
    question = "What is the capital of England?"
    result=llm_chain.run(question)


# Embeddings
model_name = "sentence-transformers/all-mpnet-base-v2"

hf = HuggingFaceEmbeddings(model_name=model_name)

hf.embed_query('this is an embedding')
hf.embed_documents(['this is an embedding','this another embedding'])


# remote hub
hf = HuggingFaceHubEmbeddings(
    repo_id=model_name,
    task="feature-extraction",
    # huggingfacehub_api_token="my-api-key",
)