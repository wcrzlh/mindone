import json
import math
from copy import deepcopy
from threading import Thread

from mindnlp.transformers import AutoProcessor
from PIL import Image
from transformers import TextIteratorStreamer

import mindspore as ms
from mindspore import Tensor, ops
from mindspore.dataset import transforms, vision

from ..idefics2.modeling_idefics2 import Idefics2VisionTransformer
from ..llama import LlamaForCausalLM, LlamaPreTrainedModel
from .configuration_minicpm import MiniCPMVConfig
from .processing_minicpmv import MiniCPMVProcessor
from .resampler import Resampler

IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)  # timm.data.IMAGENET_INCEPTION_MEAN
IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)  # timm.data.IMAGENET_INCEPTION_STD


class MiniCPMVPreTrainedModel(LlamaPreTrainedModel):
    config_class = MiniCPMVConfig


class MiniCPMV(MiniCPMVPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.llm = LlamaForCausalLM(config)
        self.vpm = self.init_vision_module()
        self.vision_dim = self.vpm.embed_dim
        self.embed_dim = self.llm.config.hidden_size
        self.resampler = self.init_resampler(self.embed_dim, self.vision_dim)
        self.transform = self.init_transform()

    def init_vision_module(self):
        # same as HuggingFaceM4/siglip-so400m-14-980-flash-attn2-navit
        model = Idefics2VisionTransformer(self.config.vision_config)
        if self.config.drop_vision_last_layer:
            model.encoder.layers = model.encoder.layers[:-1]

        setattr(model, 'embed_dim', model.embeddings.embed_dim)
        setattr(model, 'patch_size', model.embeddings.patch_size)

        return model

    def init_resampler(self, embed_dim, vision_dim):
        return Resampler(
            num_queries=self.config.query_num,
            embed_dim=embed_dim,
            num_heads=embed_dim // 128,
            kv_dim=vision_dim,
            adaptive=True
        )

    def init_transform(self):
        return transforms.Compose(
            [
                vision.ToTensor(),
                vision.Normalize(
                    mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD
                ),
            ]
        )

    def get_input_embeddings(self):
        return self.llm.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.llm.embed_tokens = value

    def get_output_embeddings(self):
        return self.llm.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.llm.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.llm = decoder

    def get_decoder(self):
        return self.llm

    def get_vllm_embedding(self, data):
        if 'vision_hidden_states' not in data:
            dtype = self.llm.model.embed_tokens.weight.dtype
            tgt_sizes = data['tgt_sizes']
            pixel_values_list = data['pixel_values']
            vision_hidden_states = []
            all_pixel_values = []
            img_cnt = []
            for pixel_values in pixel_values_list:
                img_cnt.append(len(pixel_values))
                all_pixel_values.extend([i.flatten(end_dim=1).permute(1, 0) for i in pixel_values])

            # exist image
            if all_pixel_values:
                tgt_sizes = ops.vstack(tgt_sizes).astype(ms.int32)

                if self.config.batch_vision_input:
                    max_patches = ops.max(tgt_sizes[:, 0] * tgt_sizes[:, 1])[0]

                    # FIXME how to replace torch.nn.utils.rnn.pad_sequence
                    # all_pixel_values = torch.nn.utils.rnn.pad_sequence(all_pixel_values, batch_first=True,
                    #                                                    padding_value=0.0)
                    max_length_h = max([i.shape[0] for i in all_pixel_values])
                    max_length_w = max([i.shape[1] for i in all_pixel_values])
                    for i in range(len(all_pixel_values)):
                        all_pixel_values[i] = ops.pad(all_pixel_values[i], (0, max_length_w-all_pixel_values[i].shape[1], 0, max_length_h-all_pixel_values[i].shape[0]), value=0.0)
                    all_pixel_values = ops.stack(all_pixel_values)

                    B, L, _ = all_pixel_values.shape
                    all_pixel_values = all_pixel_values.permute(0, 2, 1).reshape(B, 3, -1, L)

                    patch_attn_mask = ops.zeros(Tensor((B, 1, int(max_patches.asnumpy()))), dtype=ms.bool_)
                    for i in range(B):
                        patch_attn_mask[i, :tgt_sizes[i][0] * tgt_sizes[i][1]] = True

                    vision_embedding = self.vpm(all_pixel_values.astype(dtype),
                                                patch_attention_mask=patch_attn_mask).last_hidden_state
                    vision_embedding = self.resampler(vision_embedding, tgt_sizes)
                else:
                    # get vision_embedding foreach
                    vision_embedding = []
                    for single_tgt_size, single_pixel_values in zip(tgt_sizes, all_pixel_values):
                        single_pixel_values = single_pixel_values.unsqueeze(0)
                        B, L, _ = single_pixel_values.shape
                        single_pixel_values = single_pixel_values.permute(0, 2, 1).reshape(B, 3, -1, L)
                        single_vision_embedding = self.vpm(single_pixel_values.type(dtype)).last_hidden_state
                        single_vision_embedding = self.resampler(single_vision_embedding, single_tgt_size.unsqueeze(0))
                        vision_embedding.append(single_vision_embedding)
                    vision_embedding = ops.vstack(vision_embedding)

                start = 0
                for pixel_values in pixel_values_list:
                    img_cnt = len(pixel_values)
                    if img_cnt > 0:
                        vision_hidden_states.append(vision_embedding[start: start + img_cnt])
                        start += img_cnt
                    else:
                        vision_hidden_states.append([])
            else:  # no image
                if self.training:
                    dummy_image = ops.zeros(
                        (1, 3, 224, 224),
                        dtype=dtype
                    )
                    tgt_sizes = ms.Tensor(
                        [[(224 // self.config.patch_size), math.ceil(224 / self.config.patch_size)]]).astype(ms.int32)
                    dummy_feature = self.resampler(self.vpm(dummy_image).last_hidden_state, tgt_sizes)
                else:
                    dummy_feature = []
                for _ in range(len(pixel_values_list)):
                    vision_hidden_states.append(dummy_feature)

        else:
            vision_hidden_states = data['vision_hidden_states']

        if hasattr(self.llm.config, 'scale_emb'):
            vllm_embedding = self.llm.model.embed_tokens(data['input_ids']) * self.llm.config.scale_emb
        else:
            vllm_embedding = self.llm.model.embed_tokens(data['input_ids'])

        vision_hidden_states = [i.type(vllm_embedding.dtype) if isinstance(
            i, ms.Tensor) else i for i in vision_hidden_states]

        bs = len(data['input_ids'])
        for i in range(bs):
            cur_vs_hs = vision_hidden_states[i]
            if len(cur_vs_hs) > 0:
                cur_vllm_emb = vllm_embedding[i]
                cur_image_bound = data['image_bound'][i]
                if len(cur_image_bound) > 0:
                    image_indices = ops.stack(
                        [ops.arange(r[0], r[1], dtype=ms.int64) for r in cur_image_bound]
                    )

                    cur_vllm_emb.scatter(0, image_indices.view(-1, 1).repeat(1, cur_vllm_emb.shape[-1]),
                                          cur_vs_hs.view(-1, cur_vs_hs.shape[-1]))
                elif self.training:
                    cur_vllm_emb += cur_vs_hs[0].mean() * 0

        return vllm_embedding, vision_hidden_states

    def construct(self, data, **kwargs):
        vllm_embedding, vision_hidden_states = self.get_vllm_embedding(data)
        position_ids = data["position_ids"]
        if position_ids.dtype != ms.int64:
            position_ids = position_ids.long()

        return self.llm(
            input_ids=None,
            position_ids=position_ids,
            inputs_embeds=vllm_embedding,
            **kwargs
        )

    def _decode_text(self, result_ids, tokenizer):
        result_text = []
        for result in result_ids:
            result = result[result != 0]
            if result[0] == tokenizer.bos_id:
                result = result[1:]
            if result[-1] == tokenizer.eos_id or result[-1] == tokenizer.eot_id:
                result = result[:-1]
            result_text.append(tokenizer.decode(result).strip())
        return result_text

    def _decode(self, inputs_embeds, tokenizer, decode_text=False, **kwargs):
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        output = self.llm.generate(
            inputs_embeds=inputs_embeds,
            pad_token_id=0,
            eos_token_id=terminators,
            **kwargs
        )
        if decode_text:
            return self._decode_text(output, tokenizer)
        return output

    def _decode_stream(self, inputs_embeds, tokenizer, **kwargs):
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        streamer = TextIteratorStreamer(tokenizer=tokenizer)
        generation_kwargs = {
            'inputs_embeds': inputs_embeds,
            'pad_token_id': 0,
            'eos_token_id': terminators,
            'streamer': streamer
        }
        generation_kwargs.update(kwargs)

        thread = Thread(target=self.llm.generate, kwargs=generation_kwargs)
        thread.start()

        return streamer

    def generate(
            self,
            model_inputs,
            tokenizer=None,
            vision_hidden_states=None,
            stream=False,
            **kwargs
    ):
        bs = len(model_inputs["input_ids"])
        img_list = model_inputs["pixel_values"]
        tgt_sizes = model_inputs["tgt_sizes"]
        if img_list is None:
            img_list = [[] for i in range(bs)]
        assert bs == len(img_list)
        if vision_hidden_states is None:
            pixel_values = []
            for i in range(bs):
                img_inps = []
                for img in img_list[i]:
                    img_inps.append(img)
                if img_inps:
                    pixel_values.append(img_inps)
                else:
                    pixel_values.append([])
            model_inputs["pixel_values"] = pixel_values
            model_inputs['tgt_sizes'] = tgt_sizes
        else:
            model_inputs["vision_hidden_states"] = vision_hidden_states

        (
            input_embeds,
            vision_hidden_states,
        ) = self.get_vllm_embedding(model_inputs)

        # output_ids = self._decode(input_embeds, tokenizer, **kwargs)
        if stream:
            kwargs.pop("decode_text")
            result = self._decode_stream(input_embeds, tokenizer, **kwargs)
        else:
            result = self._decode(input_embeds, tokenizer, **kwargs)

        return result

    def chat(
            self,
            image,
            msgs,
            tokenizer,
            processor=None,
            vision_hidden_states=None,
            max_new_tokens=1024,
            sampling=True,
            max_inp_length=2048,
            system_prompt='',
            stream=False,
            **kwargs
    ):
        if processor is None:
            processor = MiniCPMVProcessor.from_pretrained(self.config._name_or_path, trust_remote_code=True)
        if isinstance(msgs, str):
            msgs = json.loads(msgs)
        copy_msgs = deepcopy(msgs)

        assert len(msgs) > 0, "msgs is empty"
        assert sampling or not stream, "if use stream mode, make sure sampling=True"

        if image is not None and isinstance(copy_msgs[0]["content"], str):
            # copy_msgs[0]['content'] = '(<image>./</image>)\n' + copy_msgs[0]['content']
            copy_msgs[0]["content"] = [image, copy_msgs[0]["content"]]

        images = []
        for i, msg in enumerate(copy_msgs):
            role = msg["role"]
            content = msg["content"]
            assert role in ["user", "assistant"]
            if i == 0:
                assert role == "user", "The role of first msg should be user"
            if isinstance(content, str):
                content = [content]
            cur_msgs = []
            for c in content:
                if isinstance(c, Image.Image):
                    images.append(c)
                    cur_msgs.append("(<image>./</image>)")
                elif isinstance(c, str):
                    cur_msgs.append(c)
            msg["content"] = "\n".join(cur_msgs)

        if system_prompt:
            sys_msg = {'role': 'system', 'content': system_prompt}
            copy_msgs = [sys_msg] + copy_msgs

        prompt = processor.tokenizer.apply_chat_template(copy_msgs, tokenize=False, add_generation_prompt=True)
        inputs = processor(prompt, images, return_tensors="ms", max_length=max_inp_length)
        # print(inputs)
        # inputs = Tensor(inputs)

        if sampling:
            generation_config = {
                "top_p": 0.8,
                "top_k": 100,
                "temperature": 0.7,
                "do_sample": True,
                "repetition_penalty": 1.05
            }
        else:
            generation_config = {
                "num_beams": 3,
                "repetition_penalty": 1.2,
            }

        generation_config.update(
            (k, kwargs[k]) for k in generation_config.keys() & kwargs.keys()
        )

        res = self.generate(
            inputs,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            vision_hidden_states=vision_hidden_states,
            stream=stream,
            decode_text=True,
            **generation_config
        )

        if stream:
            def stream_gen():
                for text in res:
                    text = text.replace(tokenizer.eot_token, '').replace(tokenizer.eos_token, '')
                    yield text

            return stream_gen()

        else:
            answer = res[0]
            return answer
