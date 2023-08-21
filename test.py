
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from modules.logging_colors import logger

generation_config = GenerationConfig.from_pretrained("/usr/models/Qwen-7B-Chat", trust_remote_code=False) # 可指定不同的生成长度、top_p等相关超参"
print(generation_config)
print("==========")
tokenizer = AutoTokenizer.from_pretrained("/usr/models/Qwen-7B-Chat",trust_remote_code=False)
logger.info(f"tokenizer.__class__.__name__ : {tokenizer.__class__.__name__}")
print(tokenizer)
print("==========")
model = AutoModelForCausalLM.from_pretrained("/usr/models/Qwen-7B-Chat", device_map="auto", trust_remote_code=False).eval()
# Specify hyperparameters for generation
print(model.generation_config)

