import base64
import copy
import functools
import json
import re
from pathlib import Path
import importlib
from functools import wraps
import traceback

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

from pydantic import BaseModel, validator

from langchain.vectorstores import FAISS

# from langchain.prompts import (
#             FewShotChatMessagePromptTemplate,
#             ChatPromptTemplate
#         )

# examples = [
#     {"input": "2+2", "output": "4"},
#     {"input": "2+3", "output": "5"},
# ]

# example_prompt = ChatPromptTemplate.from_messages(
#     [('human', '{input}'), ('ai', '{output}')]
# )

# few_shot_prompt = FewShotChatMessagePromptTemplate(
#     examples=examples,
#     # This is a prompt template used to format each individual example.
#     example_prompt=example_prompt,
# )

# final_prompt = ChatPromptTemplate.from_messages(
#     [
#         ('system', 'You are a helpful AI Assistant'),
#         few_shot_prompt,
#         ('human', '{input}'),
#     ]
# )
# final_prompt.format(input="What is 4+4?")

INIT_LLMS_DOCSTRING=r"""currently only ('text2text-generation', 'text-generation') are supported"""


def add_end_docstrings(*docstr):
    def docstring_decorator(fn):
        fn.__doc__ = (fn.__doc__ if fn.__doc__ is not None else "") + "".join(docstr)
        return fn

    return docstring_decorator


def extract_template_input_variables(template):
    """利用正则表达式提取字符串中{}内的值作为变量，列表输出
    """
    pattern = r"\{(.*?)\}"
    input_variables = re.findall(pattern, template)
    return input_variables

def string_to_dict(string):
    """字符串转字典"""
    result = {}
    lines = string.strip().split('\n')
    for line in lines:
        key, value = line.split(':')
        result[key.strip()] = value.strip()
    return result

def trace_decorator(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            final_prompt=fn(*args, **kwargs)
            trace = f"Success"
            return final_prompt,trace
        except:
            exc = traceback.format_exc()
            logger.error('Fail.')
            print(exc)
            return "",exc.replace('\n', '\n\n')
    return wrapper

@trace_decorator
def promptTemplate_to_prompt(template,
                             input_dict:str):
    """结构化prompt"""

    prompt = PromptTemplate(template=template, input_variables=extract_template_input_variables(template))
    result=prompt.format(**string_to_dict(input_dict))
    return result



# https://colab.research.google.com/drive/1h2505J5H4Y9vngzPD08ppf1ga8sWxLvZ?usp=sharing#scrollTo=wEKTalGcgxRg


@add_end_docstrings(INIT_LLMS_DOCSTRING)
def llm_chain_generate(
    prompt,
    input_dict,
    is_hub:Optional[bool] = False,
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
            "text-generation", model=shared.model, tokenizer=shared.tokenizer, max_new_tokens=10,temperature=0
        )
        local_llm = HuggingFacePipeline(pipeline=pipe)

        llm_chain = LLMChain(prompt=prompt, 
                            llm=local_llm
                            )
    
    return llm_chain.run(**string_to_dict(input_dict))
# llm_chain.run("colorful socks")


@add_end_docstrings(INIT_LLMS_DOCSTRING)
def embedding_generate(
    model_name:Optional[str] = None,
    is_hub:Optional[bool] = False,
    model_id:Optional[str] = None
):
    """
    generate the langchain embedding.
    """
    if is_hub:
        # remote hub
        embedding = HuggingFaceHubEmbeddings(
            repo_id=model_name,
            task="feature-extraction",
            # huggingfacehub_api_token="my-api-key",
        )
    else:
        # Embeddings
        embedding = HuggingFaceEmbeddings(model_name=model_name)
        embedding.embed_query('this is an embedding')
        embedding.embed_documents(['this is an embedding','this another embedding'])
    
    return embedding
