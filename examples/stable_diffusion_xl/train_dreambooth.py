import argparse
import ast
import logging
import os
import sys
import time
from functools import partial
from pathlib import Path

mindone_lib_path = os.path.abspath(os.path.abspath("../../"))
sys.path.insert(0, mindone_lib_path)

import numpy as np
from gm.data.loader import create_loader, create_loader_dreambooth  # noqa: F401
from gm.helpers import (
    SD_XL_BASE_RATIOS,
    VERSION2SPECS,
    create_model,
    get_grad_reducer,
    get_learning_rate,
    get_loss_scaler,
    get_optimizer,
    save_checkpoint,
    set_default,
)
from gm.util.util import auto_mixed_precision
from omegaconf import OmegaConf

import mindspore as ms
from mindspore import Tensor

logger = logging.getLogger(__name__)


def str2bool(b):
    if b.lower() not in ["false", "true"]:
        raise Exception("Invalid Bool Value")
    if b.lower() in ["false"]:
        return False
    return True


def get_parser_train():
    parser = argparse.ArgumentParser(description="train with sd-xl")
    parser.add_argument("--version", type=str, default="SDXL-base-1.0", choices=["SDXL-base-1.0", "SDXL-refiner-1.0"])
    parser.add_argument("--config", type=str, default="configs/training/sd_xl_base_finetune_dreambooth_lora_910b.yaml")
    parser.add_argument("--generate_class_image_config", type=str, default="configs/inference/sd_xl_base.yaml")
    parser.add_argument(
        "--task",
        type=str,
        default="txt2img",
        choices=[
            "txt2img",
        ],
    )

    parser.add_argument("--gradient_accumulation_steps", default=1, type=int, help="gradient accumulation steps")
    parser.add_argument("--clip_grad", default=False, type=ast.literal_eval, help="whether apply gradient clipping")
    parser.add_argument(
        "--max_grad_norm",
        default=1.0,
        type=float,
        help="max gradient norm for clipping, effective when `clip_grad` enabled.",
    )
    parser.add_argument("--weight", type=str, default="checkpoints/sd_xl_base_1.0_ms.ckpt")
    parser.add_argument("--per_batch_size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sd_xl_base_ratios", type=str, default="1.0")
    # parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--save_path", type=str, default="./runs")
    parser.add_argument("--save_path_with_time", type=ast.literal_eval, default=True)
    parser.add_argument("--log_interval", type=int, default=1, help="log interval")
    parser.add_argument("--save_ckpt_interval", type=int, default=1000, help="save ckpt interval")
    parser.add_argument(
        "--max_num_ckpt",
        type=int,
        default=None,
        help="Max number of ckpts saved. If exceeds, delete the oldest one. Set None: keep all ckpts.",
    )
    parser.add_argument("--data_sink", type=ast.literal_eval, default=False)
    parser.add_argument("--sink_size", type=int, default=1000)
    parser.add_argument(
        "--dataset_load_tokenizer", type=ast.literal_eval, default=True, help="create dataset with tokenizer"
    )

    # args for infer
    parser.add_argument("--infer_during_train", type=ast.literal_eval, default=False)
    parser.add_argument("--infer_interval", type=int, default=1, help="log interval")

    # args for env
    parser.add_argument("--device_target", type=str, default="Ascend", help="device target, Ascend/GPU/CPU")
    parser.add_argument(
        "--ms_mode", type=int, default=0, help="Running in GRAPH_MODE(0) or PYNATIVE_MODE(1) (default=0)"
    )
    parser.add_argument(
        "--jit_level",
        default="O2",
        type=str,
        choices=["O0", "O1", "O2"],
        help="Used to control the compilation optimization level. Supports [“O0”, “O1”, “O2”]."
        "O0: Except for optimizations that may affect functionality, all other optimizations are turned off, adopt KernelByKernel execution mode."
        "O1: Using commonly used optimizations and automatic operator fusion optimizations, adopt KernelByKernel execution mode."
        "O2: Ultimate performance optimization, adopt Sink execution mode.",
    )
    parser.add_argument("--ms_amp_level", type=str, default="O2")
    parser.add_argument(
        "--ms_enable_graph_kernel", type=ast.literal_eval, default=False, help="use enable_graph_kernel or not"
    )
    parser.add_argument("--param_fp16", type=ast.literal_eval, default=False)
    parser.add_argument("--overflow_still_update", type=ast.literal_eval, default=True)
    parser.add_argument("--max_device_memory", type=str, default=None)
    parser.add_argument("--is_parallel", type=ast.literal_eval, default=False)

    # args for ModelArts
    parser.add_argument("--enable_modelarts", type=ast.literal_eval, default=False, help="enable modelarts")
    parser.add_argument(
        "--ckpt_url", type=str, default="", help="ModelArts: obs path to pretrain model checkpoint file"
    )
    parser.add_argument("--train_url", type=str, default="", help="ModelArts: obs path to output folder")
    parser.add_argument(
        "--multi_data_url", type=str, default="", help="ModelArts: list of obs paths to multi-dataset folders"
    )
    parser.add_argument(
        "--pretrain_url", type=str, default="", help="ModelArts: list of obs paths to multi-pretrain model files"
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="/cache/pretrain_ckpt/",
        help="ModelArts: local device path to checkpoint folder",
    )

    # args for DreamBooth
    parser.add_argument(
        "--instance_data_path",
        type=str,
        default=None,
        help="Specify the folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--class_data_path",
        type=str,
        default=None,
        help="Specify the folder containing the training data of class images.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default=None,
        help="Specify the prompt with an identifier that specifies the instance.",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        help="Specify the prompt to identify images in the same class as the provided instance images.",
    )
    parser.add_argument(
        "--prior_loss_weight", type=float, default=1.0, help="Specify the weight of the prior preservation loss."
    )
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=50,
        help=(
            "Specify the number of class images for prior preservation loss. If there are not enough images"
            " already present in class_data_path, additional images will be sampled using class_prompt."
        ),
    )
    parser.add_argument(
        "--train_data_repeat",
        type=int,
        default=10,
        help=(
            "Repeat the instance images by N times in order to match the number of class images."
            " We recommend setting it as [number of class images] / [number of instance images]."
        ),
    )
    return parser


def generate_class_images(args):
    """Generate images for the class, for dreambooth"""
    class_images_dir = Path(args.class_data_path)
    if not class_images_dir.exists():
        class_images_dir.mkdir(parents=True)
    cur_class_images = len(list(class_images_dir.iterdir()))
    if cur_class_images >= args.num_class_images:
        return None

    print("Start generating class images... ")
    config = OmegaConf.load(args.generate_class_image_config)
    model, _ = create_model(
        config, checkpoints=args.weight, freeze=True, load_filter=False, amp_level=args.ms_amp_level
    )
    model.set_train(False)
    for param in model.get_parameters():
        param.requires_grad = False

    if cur_class_images < args.num_class_images:
        num_new_images = args.num_class_images - cur_class_images
        print(f"Number of class images to sample: {num_new_images}.")
    start_time = time.time()
    infer_during_train(model=model, prompt=args.class_prompt, save_path=class_images_dir, num_samples=num_new_images)

    end_time = time.time()

    print(
        f"It took {end_time-start_time:.2f} seconds to generate {num_new_images}, ",
        f"new images which are saved in: {class_images_dir}.",
    )
    del model


def train(args):
    # 1. Init Env
    args = set_default(args)

    # 2. Create LDM Engine
    config = OmegaConf.load(args.config)
    model, _ = create_model(
        config,
        checkpoints=args.weight,
        freeze=False,
        load_filter=False,
        param_fp16=args.param_fp16,
        amp_level=args.ms_amp_level,
    )
    model.model.set_train(True)  # only unet or unet+textencoder

    # 3. Create dataloader
    assert "data" in config
    per_batch_size = config.data.pop("per_batch_size")
    per_batch_size = per_batch_size if args.per_batch_size is None else args.per_batch_size
    dataloader = create_loader_dreambooth(
        instance_data_path=args.instance_data_path,
        class_data_path=args.class_data_path,
        instance_prompt=args.instance_prompt,
        class_prompt=args.class_prompt,
        per_batch_size=per_batch_size,
        rank=args.rank,
        rank_size=args.rank_size,
        train_data_repeat=args.train_data_repeat,
        tokenizer=model.conditioner.tokenize if args.dataset_load_tokenizer else None,
        token_nums=len(model.conditioner.embedders) if args.dataset_load_tokenizer else None,
        **config.data,
    )

    # 4. Create train step func
    assert "optim" in config
    lr = get_learning_rate(config.optim, config.data.total_step)
    scaler = get_loss_scaler(ms_loss_scaler="static", scale_value=1024)
    optimizer = get_optimizer(
        config.optim, lr, params=model.model.trainable_params() + model.conditioner.trainable_params()
    )
    reducer = get_grad_reducer(is_parallel=args.is_parallel, parameters=optimizer.parameters)

    if args.ms_mode == 1:
        # Pynative Mode
        train_step_fn = partial(
            model.train_step_pynative,
            grad_func=model.get_grad_func(
                optimizer, reducer, scaler, jit=True, overflow_still_update=args.overflow_still_update
            ),
        )
        model = auto_mixed_precision(model, args.ms_amp_level)
        jit_config = None
    elif args.ms_mode == 0:
        # Graph Mode
        from gm.models.trainer_factory import TrainOneStepCellDreamBooth

        model.model = auto_mixed_precision(model.model, amp_level=args.ms_amp_level)

        train_step_fn = TrainOneStepCellDreamBooth(
            model,
            optimizer,
            reducer,
            scaler,
            overflow_still_update=args.overflow_still_update,
            prior_loss_weight=args.prior_loss_weight,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            clip_grad=args.clip_grad,
            clip_norm=args.max_grad_norm,
        )

        dynamic_shape = True if "multi_aspect" in config.data.dataset_config.params.keys() else False
        if dynamic_shape:
            input_dyn = Tensor(shape=[per_batch_size, 3, None, None], dtype=ms.float32)
            token1 = Tensor(np.ones((per_batch_size, 77)), dtype=ms.int32)
            token2 = Tensor(np.ones((per_batch_size, 77)), dtype=ms.int32)
            token3 = Tensor(np.ones((per_batch_size, 2)), dtype=ms.float32)
            token4 = Tensor(np.ones((per_batch_size, 2)), dtype=ms.float32)
            token5 = Tensor(np.ones((per_batch_size, 2)), dtype=ms.float32)
            token = [token1, token2, token3, token4, token5]

            input_dyn_2 = Tensor(shape=[per_batch_size, 3, None, None], dtype=ms.float32)
            token6 = Tensor(np.ones((per_batch_size, 77)), dtype=ms.int32)
            token7 = Tensor(np.ones((per_batch_size, 77)), dtype=ms.int32)
            token8 = Tensor(np.ones((per_batch_size, 2)), dtype=ms.float32)
            token9 = Tensor(np.ones((per_batch_size, 2)), dtype=ms.float32)
            token10 = Tensor(np.ones((per_batch_size, 2)), dtype=ms.float32)
            token_reg = [token6, token7, token8, token9, token10]

            train_step_fn.set_inputs(input_dyn, input_dyn_2, *token, *token_reg)

        if model.disable_first_stage_amp:
            train_step_fn.first_stage_model.to_float(ms.float32)
        jit_config = ms.JitConfig()
    else:
        raise ValueError("args.ms_mode value must in [0, 1]")

    # 5. Start Training
    if args.max_num_ckpt is not None and args.max_num_ckpt <= 0:
        raise ValueError("args.max_num_ckpt must be None or a positive integer!")
    if args.task == "txt2img":
        train_txt2img(
            args,
            train_step_fn,
            dataloader=dataloader,
            optimizer=optimizer,
            model=model,
            jit_config=jit_config,  # for log lr  # for infer
        )
    elif args.task == "img2img":
        raise NotImplementedError
    else:
        raise ValueError(f"Unknown task {args.task}")


def train_txt2img(args, train_step_fn, dataloader, optimizer=None, model=None, **kwargs):  # for print  # for infer/ckpt
    dtype = ms.float32 if args.ms_amp_level not in ("O2", "O3") else ms.float16
    total_step = dataloader.get_dataset_size()
    loader = dataloader.create_tuple_iterator(output_numpy=True, num_epochs=1)
    s_time = time.time()

    ckpt_queue = []
    for i, data in enumerate(loader):
        # Get data, to tensor
        if not args.dataset_load_tokenizer:
            instance_data = data[0]
            class_data = data[1]
            instance_data = {k: (Tensor(v, dtype) if k != "txt" else v.tolist()) for k, v in instance_data.items()}
            class_data = {k: (Tensor(v, dtype) if k != "txt" else v.tolist()) for k, v in class_data.items()}

            # Get image and tokens
            instance_image = instance_data[model.input_key]
            instance_tokens, _ = model.conditioner.tokenize(instance_data)
            instance_tokens = [Tensor(t) for t in instance_tokens]

            class_image = class_data[model.input_key]
            class_tokens, _ = model.conditioner.tokenize(class_data)
            class_tokens = [Tensor(t) for t in class_tokens]

        else:
            assert len(data) % 2 == 0
            position = len(data) // 2
            instance_image, instance_tokens = data[0], data[1:position]
            instance_image, instance_tokens = Tensor(instance_image), [Tensor(t) for t in instance_tokens]
            class_image, class_tokens = data[position], data[position + 1 :]
            class_image, class_tokens = Tensor(class_image), [Tensor(t) for t in class_tokens]

        assert len(instance_tokens) == len(class_tokens)

        # Train a step
        if i == 0:
            print(
                "The first step will be compiled for the graph, which may take a long time; "
                "You can come back later :).",
                flush=True,
            )
        loss, overflow = train_step_fn(instance_image, class_image, *instance_tokens, *class_tokens)

        # Print meg
        if (i + 1) % args.log_interval == 0 and args.rank % 8 == 0:
            if optimizer.dynamic_lr:
                cur_lr = optimizer.learning_rate(Tensor(i, ms.int32)).asnumpy().item()
            else:
                cur_lr = optimizer.learning_rate.asnumpy().item()
            print(
                f"Step {i + 1}/{total_step}, size: {instance_image.shape[2:]}, lr: {cur_lr}, loss: {loss.asnumpy():.6f}"
                f", time cost: {(time.time()-s_time) * 1000 / args.log_interval:.2f} ms",
                flush=True,
            )
            s_time = time.time()

        # Save checkpoint
        if (i + 1) % args.save_ckpt_interval == 0 and args.rank % 8 == 0:
            model.model.set_train(False)
            save_ckpt_dir = os.path.join(args.save_path, "weights", args.version + f"_{(i+1)}.ckpt")
            save_checkpoint(
                model,
                save_ckpt_dir,
                ckpt_queue,
                args.max_num_ckpt,
                only_save_lora=False
                if not hasattr(model.model.diffusion_model, "only_save_lora")
                else model.model.diffusion_model.only_save_lora,
            )
            ckpt_queue.append(save_ckpt_dir)
            model.model.set_train(True)

        # Infer during train
        if (i + 1) % args.infer_interval == 0 and args.infer_during_train:
            print(f"Step {i + 1}/{total_step}, infer starting...")
            infer_during_train(
                model=model,
                prompt="A sks dog in a dog house.",
                save_path=os.path.join(args.save_path, "txt2img/", f"step_{i+1}_rank_{args.rank}"),
            )
            print(f"Step {i + 1}/{total_step}, infer done.", flush=True)


def infer_during_train(model, prompt, save_path, num_samples=1):
    from gm.helpers import init_sampling, perform_save_locally

    version_dict = VERSION2SPECS.get(args.version)
    W, H = SD_XL_BASE_RATIOS[args.sd_xl_base_ratios]
    C = version_dict["C"]
    F = version_dict["f"]
    is_legacy = version_dict["is_legacy"]

    value_dict = {
        "prompt": prompt,
        "negative_prompt": "",
        "orig_width": W,
        "orig_height": H,
        "target_width": W,
        "target_height": H,
        "crop_coords_top": 0,
        "crop_coords_left": 0,
        "aesthetic_score": 6.0,
        "negative_aesthetic_score": 2.5,
    }
    sampler, num_rows, num_cols = init_sampling(steps=40, num_cols=1)
    for j in range(num_samples):
        out = model.do_sample(
            sampler,
            value_dict,
            num_rows * num_cols,
            H,
            W,
            C,
            F,
            force_uc_zero_embeddings=["txt"] if not is_legacy else [],
            return_latents=False,
            filter=None,
            amp_level="O2",
        )
        perform_save_locally(save_path, out)
        print(f"{j+1}/{num_samples} image sampling done.")


if __name__ == "__main__":
    parser = get_parser_train()
    args, _ = parser.parse_known_args()

    ms.context.set_context(
        mode=args.ms_mode,
        device_target=args.device_target,
    )

    class_images_dir = Path(args.class_data_path)
    if not class_images_dir.exists():
        class_images_dir.mkdir(parents=True)
    cur_class_images = len(list(class_images_dir.iterdir()))
    if cur_class_images < args.num_class_images:
        logger.warning(f"Found {cur_class_images} class images only. The target number is {args.num_class_images}")
        generate_class_images(args)
        logger.warning(
            "Finish generating class images, please check the class images first! If the class images are ready, rerun train command to start training."
        )

    else:
        print(f"Found {cur_class_images} class images. No need to generate more class images. Start training...")
        train(args)
