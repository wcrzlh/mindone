import random
import time
from functools import partial

import numpy as np
from transformers import AutoTokenizer

from mindone.transformers import Qwen2ForCausalLM
import mindspore as ms
from mindspore import Tensor

ms.set_context(mode=0)

model_name = "/home/mikecheung/model/Qwen2.5-14B-Instruct"
model = Qwen2ForCausalLM.from_pretrained(
    model_name,
    mindspore_dtype = ms.bfloat16,
)

# infer boost
from mindspore import JitConfig
jitconfig = JitConfig(jit_level="O0", infer_boost="on")
model.set_jit_config(jitconfig)

input_ids = Tensor(shape=[1,None], dtype=ms.int32)
position_ids = Tensor(shape=[1,None], dtype=ms.int32)
attention_mask = Tensor(shape=[1,None], dtype=ms.int32)
past_key_values = None
inputs_embeds = None
labels = None
use_cache = False
output_attentions = False
output_hidden_states = False
return_dict = False
cache_position = Tensor(shape=[None,], dtype=ms.int32)
block_tables = Tensor(shape=[None, None], dtype=ms.int32)
slot_mapping = Tensor(shape=[None], dtype=ms.int32)
freqs_cis = None
mask = None
batch_valid_length = ms.mutable(Tensor(shape=[None], dtype=ms.int32))

model.set_inputs(input_ids, attention_mask, position_ids, past_key_values, inputs_embeds, labels, use_cache, output_attentions,
                 output_hidden_states, return_dict, cache_position, block_tables, slot_mapping, freqs_cis, mask, batch_valid_length)

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
    use_cache=False,
)

generated_ids = generated_ids.asnumpy()

outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

print(outputs)