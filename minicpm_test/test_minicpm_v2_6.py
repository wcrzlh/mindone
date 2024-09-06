import mindspore
from PIL import Image
from transformers import AutoTokenizer
from mindone.transformers.models.minicpm_v2_6 import MiniCPMV_v2_6

model = MiniCPMV_v2_6.from_pretrained('openbmb/MiniCPM-V-2_6', trust_remote_code=True,
    attn_implementation='eager', mindspore_dtype=mindspore.float32) # sdpa or flash_attention_2, no eager
model = model.set_train(False)
tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-V-2_6', trust_remote_code=True)

image = Image.open('./assets/airplane.jpeg').convert('RGB')

# First round chat
question = "Tell me the model of this aircraft."
msgs = [{'role': 'user', 'content': [image, question]}]

answer = model.chat(
    image=None,
    msgs=msgs,
    tokenizer=tokenizer
)
print(answer)

# Second round chat
# pass history context of multi-turn conversation
msgs.append({"role": "assistant", "content": [answer]})
msgs.append({"role": "user", "content": ["Introduce something about Airbus A380."]})

answer = model.chat(
    image=None,
    msgs=msgs,
    tokenizer=tokenizer
)
print(answer)