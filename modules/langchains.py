import base64
import copy
import functools
import json
import re
import os
from pathlib import Path
import importlib
from functools import wraps
import traceback
import torch

# import sys
# # 获取当前脚本所在的目录
# current_dir = os.path.dirname(os.path.abspath(__file__))

# # 计算项目根目录的相对路径

# project_path = os.path.join(current_dir, "experimental")

# # 将项目根目录添加到 sys.path
# sys.path.append(current_dir)
# print(current_dir)

from typing import TYPE_CHECKING, Optional, Tuple, Union, Callable, List, Any, Generator
import yaml

import modules.shared as shared
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

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

import langchain
from langchain.llms import HuggingFacePipeline, HuggingFaceHub,OpenAI
from langchain.embeddings import HuggingFaceEmbeddings,HuggingFaceHubEmbeddings
from langchain.agents import load_tools, initialize_agent, AgentType  
from langchain.agents.tools import Tool

from pydantic import BaseModel, validator
from langchain import SerpAPIWrapper,LLMMathChain,PromptTemplate, LLMChain
from langchain.vectorstores import FAISS

from langchain_experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner

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
langchain.verbose = True
os.environ['SERPAPI_API_KEY'] = '65174acf6ce3136739e1534103664bbd5fc7862b4ab41c676831e63694a943bb'

def add_end_docstrings(*docstr):
    def docstring_decorator(fn):
        fn.__doc__ = (fn.__doc__ if fn.__doc__ is not None else "") + "".join(docstr)
        return fn

    return docstring_decorator


def extract_template_input_variables(template):
    """利用正则表达式提取字符串中{}内的值作为变量，列表输出
    """
    input_variables=[]
    pattern = r"\{(.*?)\}"
    if re.search(pattern, template):
        input_variables = re.findall(pattern, template)
    return input_variables

def string_to_dict(string):
    """字符串转字典"""
    result = {}
    if re.search(r"\:", string):
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

input_variables=[]

@trace_decorator
def promptTemplate_to_prompt(question,
                             input_dict:str):
    """结构化prompt"""
    input_variables=extract_template_input_variables(question)
    prompt = PromptTemplate(template=question, input_variables=input_variables)
    result=prompt.format(**string_to_dict(input_dict))
    return result



# https://colab.research.google.com/drive/1h2505J5H4Y9vngzPD08ppf1ga8sWxLvZ?usp=sharing#scrollTo=wEKTalGcgxRg


@add_end_docstrings(INIT_LLMS_DOCSTRING)
def gengral_chain(
    question, 
    llm,
    input_dict=None, # question中的slot
):
    """
    general llm chain.
    """

    # if is_hub:
    #     llm_chain = LLMChain(prompt=prompt, 
    #                         llm=HuggingFaceHub(repo_id="google/flan-t5-xl", 
    #                                         model_kwargs={"temperature":0, 
    #                                                         "max_length":64}))
    # else:
    input_variables=extract_template_input_variables(question)
    prompt = PromptTemplate(template=question, input_variables=input_variables)
    llm_chain = LLMChain(prompt=prompt, 
                        llm=llm,
                        verbose=shared.args.langchain_verbose,
                        )
    return llm_chain.run(**string_to_dict(input_dict))


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



def loade_llm(
    state:dict, # 参数
    is_openAI:Optional[bool] = False,
):
    
    generate_params = {}
    # , 'do_sample', 'temperature', 'top_p', 'typical_p', 'repetition_penalty', 'repetition_penalty_range', 
    #           'encoder_repetition_penalty', 'top_k', 'min_length', 'no_repeat_ngram_size', 'num_beams', 'penalty_alpha', 'length_penalty', 
    #           'early_stopping', 'tfs', 'top_a', 'mirostat_mode', 'mirostat_tau', 'mirostat_eta', 'guidance_scale'
    for k in ['max_new_tokens','temperature','do_sample','top_p',]:
        generate_params[k] = state[k]

    if is_openAI:
        local_llm=OpenAI(**generate_params)
        return local_llm
    else:
        if shared.model is not None:
            pipe = pipeline(
                "text-generation", model=shared.model, tokenizer=shared.tokenizer, **generate_params
            )
            local_llm = HuggingFacePipeline(pipeline=pipe)
            return local_llm

    

# https://github.com/QwenLM/Qwen-7B/blob/main/examples/transformers_agent.md


def zero_shot_react(question,llm,):

    tools = load_tools(["serpapi", "llm-math"], llm=llm)
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=shared.args.langchain_verbose,)
    agent.run(question)

def conversational_react():
    pass

def chat_zero_shot_react(question,llm,):
    from langchain.chat_models import ChatOpenAI

    chat_model = ChatOpenAI(temperature=0)  
    tools = load_tools(["serpapi", "llm-math"], llm=llm)
    agent = initialize_agent(tools, chat_model, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=shared.args.langchain_verbose,)

def chat_conversational_react():
    pass

def react_docstore():
    from langchain import OpenAI, Wikipedia
    from langchain.agents import initialize_agent, Tool
    from langchain.agents import AgentType
    from langchain.agents.react.base import DocstoreExplorer

    docstore = DocstoreExplorer(Wikipedia())
    tools = [Tool(name="Search", func=docstore.search)] 

    llm = OpenAI(temperature=0, model_name="text-davinci-002")
    agent = initialize_agent(tools, llm, agent=AgentType.REACT_DOCSTORE, verbose=shared.args.langchain_verbose)

def plan_execute(question,llm,):
    search = SerpAPIWrapper()
    llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=shared.args.langchain_verbose)
    tools = [
        Tool(
            name = "Search",
            func=search.run,
            description="useful for when you need to answer questions about current events"
        ),
        Tool(
            name="Calculator",
            func=llm_math_chain.run,
            description="useful for when you need to answer questions about math"
        ),
    ]
    planner = load_chat_planner(llm)
    executor = load_agent_executor(llm, tools, verbose=shared.args.langchain_verbose)
    agent = PlanAndExecute(planner=planner, executor=executor, verbose=shared.args.langchain_verbose)
    agent.run(question)


LOAD_AGENT_MAP= {
    'gengral_chain': gengral_chain,
    'zero_shot_react': zero_shot_react,
    # "conversational_react": conversational_react,
    # "chat_zero_shot_react": chat_zero_shot_react,
    # "chat_conversational_react": chat_conversational_react,
    # "react_docstore": react_docstore,
    'PlanAndExecute': plan_execute,
    }

@trace_decorator
def apply_agent(typ,question,state:dict,):
    # 加载agent
    logger.info(f"Loading {typ} agent...")
    if typ not in LOAD_AGENT_MAP:
        raise ValueError(f"Invalid agent type {typ}")

    llm=loade_llm(state)
    if llm is None:
        logger.error('The llm does not exist.')
    else:
        return LOAD_AGENT_MAP[typ](question,llm)
        
