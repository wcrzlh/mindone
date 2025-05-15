import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../"))
sys.path.insert(0, mindone_lib_path)

from transformers import AutoTokenizer
from modeling_minicpm import MiniCPM3ForCausalLM
import mindspore as ms
ms.set_seed(0)

# path = 'openbmb/MiniCPM3-4B'
# tokenizer = AutoTokenizer.from_pretrained(path)
# model = MiniCPM3ForCausalLM.from_pretrained(path, mindspore_dtype=ms.bfloat16)
#
# responds, history = model.chat(tokenizer, "请写一篇关于人工智能的文章，详细介绍人工智能的未来发展和隐患。", temperature=0.7, top_p=0.7)
# print(responds)

ms.set_context(mode=0)

path = "openbmb/MiniCPM3-4B"

tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
model = MiniCPM3ForCausalLM.from_pretrained(
    path,
    mindspore_dtype=ms.bfloat16,
    _attn_implementation="flash_attention_2",
    revision="refs/pr/41")

messages = [
    {"role": "user", "content": "推荐5个北京的景点。"},
]
model_inputs = tokenizer.apply_chat_template(messages, return_tensors="np", add_generation_prompt=True)
model_inputs = ms.tensor(model_inputs)

# top_p=0.7,
# temperature=0.7

model_outputs = model.generate(
    model_inputs,
    max_new_tokens=1024,
    use_cache=False,
)

output_token_ids = [
    model_outputs[i][len(model_inputs[i]):] for i in range(len(model_inputs))
]

responses = tokenizer.batch_decode(output_token_ids, skip_special_tokens=True)[0]
print(responses)