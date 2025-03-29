import random
import time
from functools import partial

import numpy as np
from transformers import AutoTokenizer, Qwen2Config

from mindone.trainers.zero import prepare_network
from mindone.transformers import Qwen2ForCausalLM
import mindspore as ms
from mindspore import Tensor

model_name = "/home/mikecheung/model/Qwen2.5-14B-Instruct"
model = Qwen2ForCausalLM.from_pretrained(
    model_name,
    mindspore_dtype = ms.bfloat16,
    use_flash_attention_2=True
)

# infer boost
from mindspore import JitConfig
jitconfig = JitConfig(jit_level="O0", infer_boost="on")
model.set_jit_config(jitconfig)

input_ids = Tensor(shape=[1,None], dtype=ms.int32)
position_ids = Tensor(shape=[1,None], dtype=ms.int32)
attention_mask = Tensor(shape=[1,None], dtype=ms.int32)
cache_position = Tensor(shape=[None,], dtype=ms.int32)

model.set_inputs(input_ids = input_ids, position_ids=position_ids, attention_mask=attention_mask, cache_position=cache_position)

tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

input_ids = ms.Tensor(tokenizer([text], return_tensors="np").input_ids, ms.int32)
model_inputs = {}
model_inputs["input_ids"] = input_ids

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512,
    do_sample=False,
    use_cache=True,
)

generated_ids = generated_ids.asnumpy()

outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

print(outputs)