import argparse
import ast
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../../"))
sys.path.insert(0, mindone_lib_path)

from transformers import AutoTokenizer
from modeling_minicpm import MiniCPMForCausalLM
import mindspore as ms
ms.set_seed(0)

def generate(args):
    # path = 'openbmb/MiniCPM3-4B'
    # tokenizer = AutoTokenizer.from_pretrained(path)
    # model = MiniCPM3ForCausalLM.from_pretrained(path, mindspore_dtype=ms.bfloat16)
    #
    # responds, history = model.chat(tokenizer, "请写一篇关于人工智能的文章，详细介绍人工智能的未来发展和隐患。", temperature=0.7, top_p=0.7)
    # print(responds)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = MiniCPMForCausalLM.from_pretrained(
        args.model_name,
        mindspore_dtype=ms.bfloat16,
        _attn_implementation=args.attn_implementation,
        revision="refs/pr/41")

    if args.attn_implementation == "paged_attention":
        # infer boost
        from mindspore import JitConfig

        jitconfig = JitConfig(jit_level="O0", infer_boost="on")
        model.set_jit_config(jitconfig)

    messages = [
        {"role": "user", "content": args.prompt},
    ]
    model_inputs = tokenizer.apply_chat_template(messages, return_tensors="np", add_generation_prompt=True)
    model_inputs = ms.tensor(model_inputs)

    # top_p=0.7,
    # temperature=0.7

    model_outputs = model.generate(
        model_inputs,
        max_new_tokens=1024,
        use_cache=args.use_cache,
    )

    output_token_ids = [
        model_outputs[i][len(model_inputs[i]):] for i in range(len(model_inputs))
    ]

    responses = tokenizer.batch_decode(output_token_ids, skip_special_tokens=True)[0]
    print(responses)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniCPM3 demo.")

    parser.add_argument("--prompt", type=str, default="推荐5个北京的景点。")
    parser.add_argument("--model_name", type=str, default="openbmb/MiniCPM3-4B", help="Path to the pre-trained model.")
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="flash_attention_2",
        choices=["paged_attention", "flash_attention_2", "eager"],
    )
    parser.add_argument("--use_cache", type=ast.literal_eval, default=True)

    # Parse the arguments
    args = parser.parse_args()

    ms.set_context(mode=ms.GRAPH_MODE)
    generate(args)