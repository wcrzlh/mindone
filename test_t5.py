import os
import sys
import numpy as np
from transformers import AutoTokenizer

import mindspore as ms
from mindspore import ops, nn, Tensor, Parameter
from mindone.transformers.models import T5Model, T5EncoderModel, T5ForConditionalGeneration

ms.set_context(mode=0, jit_syntax_level=ms.STRICT)
# ms.set_context(mode=1,pynative_synchronize=True)
ms.set_context(deterministic="on")

# tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
# model = T5Model.from_pretrained("google-t5/t5-small")

tokenizer = AutoTokenizer.from_pretrained("DeepFloyd/t5-v1_1-xxl", revision="refs/pr/3")
model = T5EncoderModel.from_pretrained("DeepFloyd/t5-v1_1-xxl", revision="refs/pr/3", resume_download=True)

# tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
# model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")

model.set_train(False)

input_ids = tokenizer("Studies have been shown that owning a dog is good for you", return_tensors="pt").input_ids  # Batch size 1
decoder_input_ids = tokenizer("Studies show that", return_tensors="pt").input_ids  # Batch size 1

input_ids = Tensor(input_ids.numpy(), ms.int64)
decoder_input_ids = Tensor(decoder_input_ids.numpy(), ms.int64)
decoder_input_ids_ms = model._shift_right(decoder_input_ids)

# forward pass
# t5-v1.1-xxl
outputs = model(input_ids=input_ids)
# t5-small
# outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids_ms)
# flan-t5-large
# outputs = model(input_ids=input_ids, labels=decoder_input_ids)

# for encoder only
encoder_last_hidden_states_mindspore = outputs

# for encoder and decoder
# last_hidden_states_mindspore = outputs[0]
# encoder_last_hidden_states_mindspore = outputs[1]

from transformers import T5Model, T5EncoderModel, T5ForConditionalGeneration

model = T5EncoderModel.from_pretrained("DeepFloyd/t5-v1_1-xxl", revision="refs/pr/3", resume_download=True)
model.eval()

input_ids = tokenizer("Studies have been shown that owning a dog is good for you", return_tensors="pt").input_ids  # Batch size 1
decoder_input_ids = tokenizer("Studies show that", return_tensors="pt").input_ids  # Batch size 1
decoder_input_ids_torch = model._shift_right(decoder_input_ids)

# forward pass
# t5-v1.1-xxl
outputs = model(input_ids=input_ids)
# t5-small
# outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids_torch)
# flan-t5-large
# outputs = model(input_ids=input_ids, labels=labels)

# for decoder only
encoder_last_hidden_states_torch = outputs.last_hidden_state

# for encoder and decoder
# last_hidden_states_torch = outputs.last_hidden_state
# encoder_last_hidden_states_torch = outputs.encoder_last_hidden_state


# print("mean diff", np.mean(np.abs(last_hidden_states_mindspore.asnumpy()-last_hidden_states_torch.detach().numpy())))
# print("var diff", np.var(np.abs(last_hidden_states_mindspore.asnumpy()-last_hidden_states_torch.detach().numpy())))

print("encoder mean diff", np.mean(np.abs(encoder_last_hidden_states_mindspore.asnumpy()-encoder_last_hidden_states_torch.detach().numpy())))
print("encoder var diff", np.var(np.abs(encoder_last_hidden_states_mindspore.asnumpy()-encoder_last_hidden_states_torch.detach().numpy())))
