import os
import sys
import mindspore as ms
from mindspore import nn, ops, Parameter, Tensor
# import deepspeed
from mindnlp.engine import Trainer
from collections.abc import Mapping
from transformers.trainer_pt_utils import nested_detach
# from mindnlp.transformers.utils import is_sagemaker_mp_enabled
# from transformers.integrations import is_deepspeed_zero3_enabled
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

mindone_lib_path = os.path.abspath(os.path.abspath("../../../"))
sys.path.insert(0, mindone_lib_path)

from mindone.transformers.modeling_utils import MSPreTrainedModel

from transformers.utils import logging

logger = logging.get_logger(__name__)

def nested_detach(tensors):
    "Detach `tensors` (even if it's a nested list/tuple/dict of tensors)."
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_detach(t) for t in tensors)
    elif isinstance(tensors, Mapping):
        return type(tensors)({k: nested_detach(t) for k, t in tensors.items()})
    return tensors.detach() if isinstance(tensors, ms.Tensor) else tensors


class CPMTrainer(Trainer):
    def __init__(self, reducer=None):
        super().__init__()
        self.reducer = reducer

    def compute_loss(self, model, inputs, return_outputs=False):
        if "labels" in inputs:
            labels = inputs["labels"]
        else:
            labels = None
        
        if not self.args.use_lora:
            outputs = self.model(data = inputs, use_cache=False)
        else:
            with self.model._enable_peft_forward_hooks(**inputs):
                outputs = self.model.base_model(data = inputs, use_cache=False)
                
        if labels is not None:
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            logits = outputs.logits.view(-1, self.model.config.vocab_size).astype(ms.float32)
            labels = labels.view(-1)
            # Enable model parallelism
            loss = loss_fct(logits, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

    def prediction_step(
        self,
        model: nn.Cell,
        inputs: Dict[str, Union[ms.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[ms.Tensor], Optional[ms.Tensor], Optional[ms.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Cell`):
                The model to evaluate.
            inputs (`Dict[str, Union[ms.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[ms.Tensor], Optional[ms.Tensor], Optional[ms.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        has_labels = (
            False
            if len(self.label_names) == 0
            else all(inputs.get(k) is not None for k in self.label_names)
        )
        # For CLIP-like models capable of returning loss values.
        # If `return_loss` is not specified or being `None` in `inputs`, we check if the default value of `return_loss`
        # is `True` in `model.forward`.
        return_loss = inputs.get("return_loss", None)
        if return_loss is None:
            return_loss = self.can_return_loss
        loss_without_labels = (
            True if len(self.label_names) == 0 and return_loss else False
        )

        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(
                    self.model.config, "keys_to_ignore_at_inference", []
                )
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels or loss_without_labels:
            labels = nested_detach(tuple(inputs.get(name)
                                   for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with ms._no_grad():
            # if is_sagemaker_mp_enabled():
            #     raw_outputs = smp_forward_only(model, inputs)
            #     if has_labels or loss_without_labels:
            #         if isinstance(raw_outputs, dict):
            #             loss_mb = raw_outputs["loss"]
            #             logits_mb = tuple(
            #                 v
            #                 for k, v in raw_outputs.items()
            #                 if k not in ignore_keys + ["loss"]
            #             )
            #         else:
            #             loss_mb = raw_outputs[0]
            #             logits_mb = raw_outputs[1:]
            #
            #         loss = loss_mb.reduce_mean().detach().cpu()
            #         logits = smp_nested_concat(logits_mb)
            #     else:
            #         loss = None
            #         if isinstance(raw_outputs, dict):
            #             logits_mb = tuple(
            #                 v for k, v in raw_outputs.items() if k not in ignore_keys
            #             )
            #         else:
            #             logits_mb = raw_outputs
            #         logits = smp_nested_concat(logits_mb)
            # else:
            if has_labels or loss_without_labels:
                with self.compute_loss_context_manager():
                    loss, outputs = self.compute_loss(
                        model, inputs, return_outputs=True
                    )
                loss = loss.mean().detach()

                if isinstance(outputs, dict):
                    logits = tuple(
                        v
                        for k, v in outputs.items()
                        if k not in ignore_keys + ["loss"]
                    )
                else:
                    logits = outputs[1:]
            else:
                loss = None
                with self.compute_loss_context_manager():
                    outputs = model(**inputs)
                if isinstance(outputs, dict):
                    logits = tuple(
                        v for k, v in outputs.items() if k not in ignore_keys
                    )
                else:
                    logits = outputs
                # TODO: this needs to be fixed and made cleaner later.
                if self.args.past_index >= 0:
                    self._past = outputs[self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        return (loss, logits, labels)
        
    def training_step(self, model: nn.Cell, inputs: Dict[str, Union[ms.Tensor, Any]]) -> ms.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Cell`):
                The model to train.
            inputs (`Dict[str, Union[ms.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `ms.Tensor`: The tensor with training loss on this batch.
        """
        model.set_train()
        inputs = self._prepare_inputs(inputs)

        # if is_sagemaker_mp_enabled():
        #     loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
        #     return loss_mb.reduce_mean().detach().to(self.args.device)

        # with self.compute_loss_context_manager():
        #     loss = self.compute_loss(model, inputs)

        # del inputs

        # if self.args.n_gpu > 1:
        #     loss = loss.mean()  # mean() to average on multi-gpu parallel training

        # if self.use_apex:
        #     with amp.scale_loss(loss, self.optimizer) as scaled_loss:
        #         scaled_loss.backward()
        # else:
        #     self.accelerator.backward(loss)

        def forward(inputs):
            return self.compute_loss(model, inputs)

        # if getattr(self, 'grad_fn', None) is None or self.model_reload:
        self.grad_fn = ops.value_and_grad(forward, None, self.optimizer.parameters)

        loss, grads = self.grad_fn(inputs)

        if self.args.distributed:
            grads = self.reducer(grads)

        del inputs

        return loss / self.args.gradient_accumulation_steps, grads
    
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        supported_classes = (MSPreTrainedModel,)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, supported_classes):
            if state_dict is None:
                state_dict = self.model.state_dict()
            # if isinstance(unwrap_model(self.model), supported_classes):
            #     unwrap_model(self.model).save_pretrained(
            #         output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
            #     )
            # else:
            #     logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
            #     if self.args.save_safetensors:
            #         safetensors.torch.save_file(
            #             state_dict, os.path.join(output_dir, SAFE_WEIGHTS_NAME), metadata={"format": "pt"}
            #         )
            #     else:
            ms.save_checkpoint(state_dict, os.path.join(output_dir, "minicpm.ckpt"))
        else:
            
            self.model.save_pretrained(
                output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
            )

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        # ms.save_checkpoint(self.args, os.path.join(output_dir, "training_args.ckpt"))
