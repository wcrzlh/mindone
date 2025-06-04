#    Copyright 2024 Hao Zhang
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union, Dict
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaModel, LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput
from longva.longva.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from mindone.transformers import Qwen2Model, Qwen2ForCausalLM
from transformers import Qwen2Config
import pdb
import time
import random

random.seed(42)
import torch
from statistics import mean
import torch.nn.functional as F
import PIL
from decord import VideoReader, cpu
from .conversation import conv_templates, SeparatorStyle
from .constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, \
    DEFAULT_IM_END_TOKEN, DEFAULT_IMAGE_TOKEN
from .mm_utils import tokenizer_image_token, load_video


class LlavaQwenConfig(Qwen2Config):
    model_type = "llava_qwen"


class LlavaQwenModel(LlavaMetaModel, Qwen2Model):
    config_class = LlavaQwenConfig

    def __init__(self, config: Qwen2Config):
        super(LlavaQwenModel, self).__init__(config)


class LlavaQwenForCausalLM(Qwen2ForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaQwenConfig

    def __init__(self, config):
        # super(Qwen2ForCausalLM, self).__init__(config)
        Qwen2ForCausalLM.__init__(self, config)
        config.model_type = "llava_qwen"
        config.rope_scaling = None

        self.model = LlavaQwenModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def uniform_sampling(self, embeds, start_idx, end_idx, step):
        indices = torch.arange(start_idx, end_idx, step).to(device=embeds.device)
        return embeds.index_select(1, indices), indices

    def pooling_sampling(self, embeds, start_idx, end_idx, step, pool_type='avg'):
        selected = embeds[:, start_idx:end_idx, :]
        B, D, L = selected.shape
        kernel_size = step
        stride = step

        selected_transposed = selected.transpose(1, 2)  # shape: (1, 12, 4)

        if pool_type == 'avg_pool':
            pooled = F.avg_pool1d(selected_transposed, kernel_size=kernel_size, stride=stride)
        elif pool_type == 'max_pool':
            pooled = F.max_pool1d(selected_transposed, kernel_size=kernel_size, stride=stride)
        else:
            raise ValueError(f"Unsupported pooling type: {pool_type}")

        pooled = pooled.transpose(1, 2)  # shape: (1, 2, 12)
        return pooled, torch.arange(start_idx, start_idx + pooled.shape[1] * step, step).to(device=embeds.device)

    def process_block(self, block_embeds, current_past_key_values=None, bsz=1, device=None, position_ids=None,
                      key_position_ids=None):
        if current_past_key_values is None:
            seq_len = block_embeds.size(1)
            position_ids = torch.arange(0, seq_len, device=device).expand(bsz, -1)
            attention_mask = torch.ones((bsz, seq_len), device=device, dtype=torch.long)
        else:
            seq_len = block_embeds.size(1)
            prefix_len = current_past_key_values[0][0].size(2)
            attention_mask = torch.ones((bsz, prefix_len + seq_len), device=device, dtype=torch.long)

        outputs = self.model(
            inputs_embeds=block_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            key_position_ids=key_position_ids,
            past_key_values=current_past_key_values,
            use_cache=True,
            return_dict=True,
        )
        return outputs.past_key_values

    def pooling_kvs(self, kvs, step):
        # kvs shape: (bsz, 4, seq_len, head_dim)
        kernel_size = step
        stride = step
        # kvs = kvs.transpose(2, 3)
        # pooled_kvs = F.avg_pool1d(kvs, kernel_size=kernel_size, stride=stride)
        kvs_permuted = kvs.permute(0, 1, 3, 2)  # (batch_size, num_heads, feature_dim, sequence_length)
        N_flat = kvs_permuted.shape[0] * kvs_permuted.shape[1]
        C = kvs_permuted.shape[2]
        L = kvs_permuted.shape[3]
        kvs_for_pool = kvs_permuted.reshape(N_flat, C, L)
        pooled_kvs = F.avg_pool1d(kvs_for_pool, kernel_size=kernel_size, stride=stride)
        pooled_kvs_restored = pooled_kvs.view(kvs.shape[0], kvs.shape[1], pooled_kvs.shape[1],
                                              pooled_kvs.shape[2]).permute(0, 1, 3, 2)
        return pooled_kvs_restored

    def get_sparse_attention_mask(self, total_len, num_blocks, block_size, time_token_start_indices,
                                  time_token_end_indices, time_token_indices, visual_token_start_pos,
                                  visual_token_end_pos, attention_mask, inputs_embeds, prev_blocks_num=None):

        causal_mask = torch.tril(torch.ones((total_len, total_len), dtype=torch.bool)).unsqueeze(0).repeat(1, 1, 1)
        mask = torch.zeros(total_len, total_len, dtype=torch.bool)
        start = visual_token_start_pos

        record_block_start = []
        for i in range(num_blocks):
            next_time_token_pos = (i + 1) * block_size
            if next_time_token_pos >= len(time_token_start_indices):
                end = visual_token_end_pos
            else:
                end = time_token_start_indices[next_time_token_pos]

            mask[start:end, start:end] = True

            if len(record_block_start) >= prev_blocks_num:
                prev_start = record_block_start[-prev_blocks_num]
            else:
                prev_start = visual_token_start_pos

            mask[start:end, prev_start:start] = True
            record_block_start.append(start)
            start = end

        mask[:, :visual_token_start_pos] = True
        mask[visual_token_end_pos:, :] = True

        for idx in time_token_indices:
            mask[idx, :] = True
            mask[:, idx] = True

        causal_mask = torch.tril(torch.ones(total_len, total_len, dtype=torch.bool))
        final_mask = (mask & causal_mask).unsqueeze(0).unsqueeze(0).to(dtype=attention_mask.dtype,
                                                                       device=attention_mask.device)

        num_allowed = final_mask.sum().item()
        upper_triangle_num = total_len * (total_len + 1) // 2
        ratio = num_allowed / upper_triangle_num

        invert_mask = 1.0 - final_mask
        final_mask = ((1.0 - final_mask) * -1e9).to(dtype=inputs_embeds.dtype)
        return final_mask, ratio

    def cat_history_kvs(self, prefix_kvs, kvs_part2, kvs_part3):
        prefix_kvs = [[kvs] for kvs in prefix_kvs]
        cat_kvs = []
        for prefix_kvs_this_layer, kvs_part2_this_layer, kvs_part3_this_layer in zip(prefix_kvs, kvs_part2, kvs_part3):
            prefix_key_this_layer = [tmp[0] for tmp in prefix_kvs_this_layer]
            prefix_val_this_layer = [tmp[1] for tmp in prefix_kvs_this_layer]

            key_part2_this_layer = [tmp[0] for tmp in kvs_part2_this_layer]
            val_part2_this_layer = [tmp[1] for tmp in kvs_part2_this_layer]

            key_part3_this_layer = [tmp[0] for tmp in kvs_part3_this_layer]
            val_part3_this_layer = [tmp[1] for tmp in kvs_part3_this_layer]

            key_this_layer = torch.cat(prefix_key_this_layer + key_part2_this_layer + key_part3_this_layer, dim=-2)
            val_this_layer = torch.cat(prefix_val_this_layer + val_part2_this_layer + val_part3_this_layer, dim=-2)

            cat_kvs.append((key_this_layer, val_this_layer))
        return cat_kvs

    def forward_streaming(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            key_position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            dpo_forward: Optional[bool] = False,
            cache_position=None,
            visual_token_start_pos=None,
            visual_token_end_pos=None,
            time_token_start_indices=None,
            frames_num=None,
            time_token_indices=None,
            time_token_end_indices=None,
            block_size_chosed=None,
            prev_blocks_num=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        block_size = block_size_chosed
        visual_token_start_pos = visual_token_start_pos
        visual_token_end_pos = visual_token_end_pos
        visual_len = visual_token_end_pos - visual_token_start_pos
        num_blocks = (frames_num + block_size * 4 - 1) // (block_size * 4)
        # print(f'block_size: {block_size}, num_blocks: {num_blocks}')

        # streaming inps
        blocks_positions = [[(0, 0, visual_token_start_pos)]]
        frames_groups = [(0, visual_token_start_pos)]
        for idx, (time_start, time_end) in enumerate(zip(time_token_start_indices, time_token_end_indices)):
            if idx + 1 < len(time_token_start_indices):
                frames_group_end = time_token_start_indices[idx + 1]
            else:
                frames_group_end = visual_token_end_pos
            frames_groups.append(
                (time_start, time_end, frames_group_end)
            )

        single_block = []
        for group in frames_groups[1:]:
            single_block.append(group)
            if len(single_block) == block_size:
                blocks_positions.append(single_block)
                single_block = []
        if len(single_block) != 0:
            blocks_positions.append(single_block)
        num_blocks = len(blocks_positions)

        start = time.time()
        record_prefill_time = 0

        full_inputs_embeds = inputs_embeds
        bsz, total_len, embed_dim = full_inputs_embeds.size()
        device = full_inputs_embeds.device

        prefix_embeds = full_inputs_embeds[:, :visual_token_start_pos, :]
        visual_embeds = full_inputs_embeds[:, visual_token_start_pos:visual_token_end_pos, :]
        suffix_embeds = full_inputs_embeds[:, visual_token_end_pos:, :]
        num_visual_tokens = visual_embeds.size(1)

        all_past_key_values = [[] for _ in range(len(self.model.layers))]  # 假设 model 有 layers 属性
        prefix_past_key_values = []

        torch.cuda.reset_peak_memory_stats()

        if prefix_embeds.size(1) > 0:
            pkv = self.process_block(prefix_embeds, bsz=bsz, device=device)
            for i in range(len(pkv)):
                all_past_key_values[i].append(pkv[i])
                prefix_past_key_values.append(pkv[i])

        prev_blocks = blocks_positions[1:1 + prev_blocks_num]
        prev_the_first_block = prev_blocks[0]
        prev_b_start = prev_the_first_block[0][0]
        prev_the_last_block = prev_blocks[-1]
        prev_b_end = prev_the_last_block[-1][-1]

        block_streaming_past_key_values = prefix_past_key_values

        query_position_ids = torch.arange(prev_b_start, prev_b_end, dtype=torch.long, device=device)
        past_key_position_ids = torch.arange(0, block_streaming_past_key_values[0][0].size(2), dtype=torch.long,
                                             device=device)
        key_position_ids = torch.cat([past_key_position_ids, query_position_ids], dim=0)

        visual_embeds_this_block = full_inputs_embeds[:, prev_b_start:prev_b_end, :]
        pkv = self.process_block(visual_embeds_this_block, current_past_key_values=block_streaming_past_key_values,
                                 bsz=bsz, device=device, position_ids=query_position_ids.unsqueeze(0),
                                 key_position_ids=key_position_ids.unsqueeze(0))

        for i in range(len(pkv)):
            for block in prev_blocks:
                block_start, _, _ = block[0]
                _, _, block_end = block[-1]
                all_past_key_values[i].append(
                    (pkv[i][0][:, :, block_start:block_end], pkv[i][1][:, :, block_start:block_end]))

        block_streaming_past_key_values_part1 = prefix_past_key_values
        position_ids_part1 = torch.arange(0, prefix_past_key_values[0][0].size(2), dtype=torch.long, device=device)
        block_streaming_past_key_values_part2 = [[] for _ in range(len(self.model.layers))]  # 存
        position_ids_part2 = torch.tensor([], dtype=torch.long, device=device)
        block_streaming_past_key_values_part3 = None
        position_ids_part3 = None

        query_position_ids = None
        for idx, single_block in enumerate(blocks_positions[:]):
            if idx == 0:
                continue
            if idx <= prev_blocks_num:
                continue

            b_start, _, _ = single_block[0]
            _, _, b_end = single_block[-1]
            visual_embeds_this_block = full_inputs_embeds[:, b_start:b_end, :]
            prev_blocks = blocks_positions[max(idx - prev_blocks_num, 1):idx]
            prev_the_first_block = prev_blocks[0]
            prev_b_start = prev_the_first_block[0][0]

            this_block_length = b_end - prev_b_start
            prev_block_length = b_start - prev_b_start
            true_block_length = b_end - b_start

            block_streaming_past_key_values_part3 = [tmp[-prev_blocks_num:] for tmp in all_past_key_values]
            # block_streaming_past_key_values_part3 = [
            #     [
            #         (t[0].to(device=device), t[1].to(device=device))
            #         for t in sublist
            #     ]
            #     for sublist in block_streaming_past_key_values_part3
            # ]

            block_streaming_past_key_values = self.cat_history_kvs(block_streaming_past_key_values_part1,
                                                                   block_streaming_past_key_values_part2,
                                                                   block_streaming_past_key_values_part3)

            query_position_ids = torch.arange(b_start, b_end, dtype=torch.long, device=device)
            position_ids_part3 = torch.arange(prev_b_start, b_start, dtype=torch.long, device=device)
            key_position_ids = torch.cat(
                [position_ids_part1, position_ids_part2, position_ids_part3, query_position_ids], dim=0)

            start_1 = time.time()
            pkv = self.process_block(visual_embeds_this_block, current_past_key_values=block_streaming_past_key_values,
                                     bsz=bsz, device=device, position_ids=query_position_ids.unsqueeze(0),
                                     key_position_ids=key_position_ids.unsqueeze(0))
            end_1 = time.time()

            record_prefill_time += end_1 - start_1

            for i in range(len(pkv)):
                length_before_chunk = block_streaming_past_key_values[i][0].size(2)
                key_this_block, val_this_block = pkv[i]
                key_this_block = key_this_block[:, :, length_before_chunk:, :]
                val_this_block = val_this_block[:, :, length_before_chunk:, :]
                all_past_key_values[i].append((key_this_block, val_this_block))
                # all_past_key_values[i].append( (key_this_block.to('cpu'), val_this_block.to('cpu')) )

                time_keys_list = []
                time_vals_list = []

                extract_timestamps_position_ids_list = []
                for group in prev_the_first_block:
                    time_start, time_end, _ = group
                    extract_timestamps_position_ids_list.append(
                        torch.arange(time_start, time_end, dtype=torch.long, device=device))

                    time_start = time_start - prev_b_start
                    time_end = time_end - prev_b_start

                    time_keys_list.append(block_streaming_past_key_values_part3[i][0][0][:, :, time_start:time_end, :])
                    time_vals_list.append(block_streaming_past_key_values_part3[i][0][1][:, :, time_start:time_end, :])

                time_keys = torch.cat(time_keys_list, dim=2)
                time_vals = torch.cat(time_vals_list, dim=2)

                block_streaming_past_key_values_part2[i].append((time_keys, time_vals))

                if i == 0:
                    position_ids_part2 = torch.cat([position_ids_part2] + extract_timestamps_position_ids_list, dim=0)

        merged_pkv = []
        for layer_pkvs in all_past_key_values:
            if not layer_pkvs:
                continue
            keys = torch.cat([pkv[0].to(device=device) for pkv in layer_pkvs], dim=2)  # dim=2 是 sequence 维度
            values = torch.cat([pkv[1].to(device=device) for pkv in layer_pkvs], dim=2)
            merged_pkv.append((keys, values))

        pkv = merged_pkv
        del block_streaming_past_key_values
        del all_past_key_values
        del block_streaming_past_key_values_part1
        del block_streaming_past_key_values_part2
        del block_streaming_past_key_values_part3
        torch.cuda.empty_cache()

        # TODO: bi-decoding acceleration
        mixed_prefill_past_key_values = pkv
        prefill_len = visual_token_end_pos

        # Process suffix
        if suffix_embeds.size(1) > 0:
            seq_len = suffix_embeds.size(1)
            total_len = prefill_len + seq_len
            position_ids = torch.arange(prefill_len, total_len, device=device, dtype=torch.long).expand(bsz, -1)
            key_position_ids = torch.arange(0, total_len, device=device, dtype=torch.long).expand(bsz, -1)
            attention_mask = torch.ones((bsz, total_len), device=device, dtype=torch.long)

            outputs = super().forward(
                inputs_embeds=suffix_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                key_position_ids=key_position_ids,
                past_key_values=mixed_prefill_past_key_values,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                use_cache=True,
                return_dict=return_dict,
                # blocks_positions=None,
            )
            del mixed_prefill_past_key_values
            torch.cuda.empty_cache()

        return outputs

    def forward_mask(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            dpo_forward: Optional[bool] = False,
            cache_position=None,
            visual_token_start_pos=None,
            visual_token_end_pos=None,
            time_token_start_indices=None,
            time_token_end_indices=None,
            frames_num=None,
            time_token_indices=None,
            prev_blocks_num=None,
            block_size_chosed=None
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        bsz, total_len, embed_dim = inputs_embeds.size()
        visual_token_start_pos = visual_token_start_pos
        visual_token_end_pos = visual_token_end_pos
        visual_len = visual_token_end_pos - visual_token_start_pos

        block_size_list = [2, 4, 8, 16, 32]
        best_block_size = None
        min_diff = float('inf')

        block_size = block_size_chosed
        num_blocks = (frames_num + block_size * 4 - 1) // (block_size * 4)
        final_mask, ratio = self.get_sparse_attention_mask(total_len, num_blocks, block_size, time_token_start_indices,
                                                           time_token_end_indices, time_token_indices,
                                                           visual_token_start_pos, visual_token_end_pos, attention_mask,
                                                           inputs_embeds, prev_blocks_num)

        # print(f'frames:{frames_num}, block_num:{num_blocks}, bsz:{block_size}, prev_blocks_num:{prev_blocks_num}, ratio:{ratio}')

        return super().forward(
            input_ids=input_ids,
            attention_mask=final_mask,  # final_mask
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            key_position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            images: Optional[torch.FloatTensor] = None,
            image_sizes: Optional[List[List[int]]] = None,
            return_dict: Optional[bool] = None,
            modalities: Optional[List[str]] = ["image"],
            dpo_forward: Optional[bool] = False,
            cache_position=None,
            time_embedding=None,
            visual_token_start_pos=None,
            visual_token_end_pos=None,
            time_token_start_indices=None,
            frames_num=None,
            time_token_indices=None,
            time_token_end_indices=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if input_ids is not None and input_ids.size(1) == 1:
            past_key_len = past_key_values[0][0].size(-2)
            key_position_ids = torch.arange(0, past_key_len + 1, device=position_ids.device, dtype=torch.long).expand(1,
                                                                                                                      -1)
            if position_ids[0][0] != past_key_len:
                position_ids = torch.tensor([[past_key_len]]).to(device=position_ids.device, dtype=position_ids.dtype)
                key_position_ids = torch.arange(0, past_key_len + 1, device=position_ids.device,
                                                dtype=torch.long).expand(1, -1)

            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                key_position_ids=key_position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        if inputs_embeds is None:
            (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds,
             labels) = self.prepare_inputs_labels_for_multimodal(input_ids, position_ids, attention_mask,
                                                                 past_key_values, labels, images, modalities,
                                                                 image_sizes, time_embedding)

        if self.config.enable_sparse:
            block_size_chosed = self.config.sparse_config['block_size_chosed']
            prev_blocks_num = self.config.sparse_config['prev_blocks_num']
            if self.config.sparse_mode == 'streaming':
                return self.forward_streaming(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    key_position_ids=key_position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    labels=labels,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    cache_position=cache_position,
                    visual_token_start_pos=visual_token_start_pos,
                    visual_token_end_pos=visual_token_end_pos,
                    time_token_start_indices=time_token_start_indices,
                    frames_num=frames_num,
                    time_token_indices=time_token_indices,
                    time_token_end_indices=time_token_end_indices,
                    block_size_chosed=block_size_chosed,
                    prev_blocks_num=prev_blocks_num,
                )
            elif self.config.sparse_mode == 'mask':
                return self.forward_mask(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    labels=labels,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    cache_position=cache_position,
                    visual_token_start_pos=visual_token_start_pos,
                    visual_token_end_pos=visual_token_end_pos,
                    time_token_start_indices=time_token_start_indices,
                    frames_num=frames_num,
                    time_token_indices=time_token_indices,
                    time_token_end_indices=time_token_end_indices,
                    block_size_chosed=block_size_chosed,
                    prev_blocks_num=prev_blocks_num,
                )
        else:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

    @torch.no_grad()
    def generate(
            self,
            inputs: Optional[torch.Tensor] = None,
            images: Optional[torch.Tensor] = None,
            image_sizes: Optional[torch.Tensor] = None,
            modalities: Optional[List[str]] = ["image"],
            time_embedding=None,
            **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:

        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)

        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None and images[0].size(0) > 0:
            IMAGE_TOKEN_INDEX = -200
            TOKEN_PERFRAME = 36
            frames_num = images[0].size(0)
            visual_token_start_pos = (inputs == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[1].item()
            num_tokens = time_embedding[0].size(0)
            visual_token_end_pos = visual_token_start_pos + num_tokens
            kwargs['visual_token_start_pos'] = visual_token_start_pos
            kwargs['visual_token_end_pos'] = visual_token_end_pos
            # time_token_start_indices = (time_embedding[0] == 1462).nonzero(as_tuple=True)
            time_token_start_indices = (time_embedding[0] == 1462).nonzero(as_tuple=True)[0].cpu().tolist()
            kwargs['time_token_start_indices'] = [idx + visual_token_start_pos for idx in time_token_start_indices]
            # kwargs['time_token_start_indices'] = time_token_start_indices + visual_token_start_pos
            kwargs['frames_num'] = frames_num
            time_token_indices = (time_embedding[0] != 151654).nonzero(as_tuple=True)[0].cpu().tolist()
            kwargs['time_token_indices'] = [idx + visual_token_start_pos for idx in time_token_indices]
            time_token_end_indices = (time_embedding[0] == 25).nonzero(as_tuple=True)[0].cpu().tolist()
            kwargs['time_token_end_indices'] = [idx + visual_token_start_pos + 1 for idx in time_token_end_indices]
            # kwargs['time_token_end_indices'] = time_token_end_indices + visual_token_start_pos

        # print(images[0].shape)
        if images is not None:
            (inputs, position_ids, attention_mask, _, inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal(
                inputs, position_ids, attention_mask, None, None, images, modalities, image_sizes=image_sizes,
                time_embedding=time_embedding)

        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        # print(inputs_embeds.shape)
        return super().generate(position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds,
                                **kwargs)

    @torch.no_grad()
    def chat(self,
             video_path,
             tokenizer,
             user_prompt,
             chat_history=None,
             return_history=True,
             max_num_frames=512,
             sample_fps=1,
             max_sample_fps=4,
             generation_config={}):

        # prepare text input
        conv = conv_templates["qwen_1_5"].copy()
        if chat_history is None or len(chat_history) == 0:
            user_prompt = f'{DEFAULT_IMAGE_TOKEN}\n{user_prompt}'
        else:
            assert DEFAULT_IMAGE_TOKEN in chat_history[0]['content'], chat_history
            for msg in chat_history:
                conv.append_message(msg['role'], msg['content'])

        conv.append_message(conv.roles[0], user_prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(
            self.model.device)

        # prepare video input
        frames, timestamps = load_video(video_path, max_num_frames, fps=sample_fps, max_fps=max_sample_fps)

        time_stamps = []
        token_frames_sum = (len(timestamps) + 3) // 4
        compress_frame = timestamps[::4]
        time_embedding = []
        for time in compress_frame:
            item = f"Time {time}s:"
            time_embedding.append(tokenizer(item).input_ids)
            time_embedding.append([151654] * 144)

        time_embedding = [item for sublist in time_embedding for item in sublist]
        time_embedding = torch.tensor(time_embedding, dtype=torch.long).to(self.model.device)
        time_stamps.append(time_embedding)

        video_tensor = self.get_vision_tower().image_processor.preprocess(frames, return_tensors="pt")[
            "pixel_values"].to(self.model.device, dtype=torch.float16)

        with torch.inference_mode():
            output_ids = self.generate(input_ids, images=[video_tensor], time_embedding=time_stamps,
                                       modalities=["video"], **generation_config)

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        if chat_history is None:
            chat_history = []

        chat_history.append({"role": conv.roles[0], "content": user_prompt})
        chat_history.append({"role": conv.roles[1], "content": outputs})
        if return_history:
            return outputs, chat_history
        else:
            return outputs

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        visual_token_start_pos = kwargs.get("visual_token_start_pos", None)
        visual_token_end_pos = kwargs.get("visual_token_end_pos", None)
        time_token_start_indices = kwargs.get("time_token_start_indices", None)
        frames_num = kwargs.get("frames_num", None)
        time_token_indices = kwargs.get("time_token_indices", None)
        time_token_end_indices = kwargs.get("time_token_end_indices", None)

        inputs = super().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values,
                                                       inputs_embeds=inputs_embeds, **kwargs)

        inputs["visual_token_start_pos"] = visual_token_start_pos
        inputs["visual_token_end_pos"] = visual_token_end_pos
        inputs["time_token_start_indices"] = time_token_start_indices
        inputs["frames_num"] = frames_num
        inputs["time_token_indices"] = time_token_indices
        inputs["time_token_end_indices"] = time_token_end_indices

        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        return inputs


AutoConfig.register("llava_qwen", LlavaQwenConfig)
AutoModelForCausalLM.register(LlavaQwenConfig, LlavaQwenConfig)