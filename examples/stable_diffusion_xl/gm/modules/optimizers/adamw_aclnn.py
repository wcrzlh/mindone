from __future__ import absolute_import

import mindspore as ms
from mindspore import _checkparam as validator
from mindspore import ops
from mindspore.nn import Optimizer
from mindspore.ops import auto_generate as gen

_optim_adamw_opt = ops.MultitypeFuncGraph("optim_adamw_opt")
hyper_map = ops.HyperMap()


@_optim_adamw_opt.register(
    "Function",
    "Float",
    "Float",
    "Float",
    "Tensor",
    "Bool",
    "Bool",
    "Tensor",
    "Tensor",
    "Tensor",
    "Tensor",
    "Tensor",
    "Tensor",
)
def _run_optim_adamw_opt(
    opt, beta1, beta2, eps, step, asmgrad, maximize, learning_rate, weight_decay, parameters, grads, exp_avg, exp_avg_sq
):
    success = True
    max_exp_avg_sq = ops.zeros_like(exp_avg)
    learning_rate = float(learning_rate)
    weight_decay = float(weight_decay)
    opt(
        parameters,
        exp_avg,
        exp_avg_sq,
        max_exp_avg_sq,
        grads.astype(parameters.dtype),
        step,
        learning_rate,
        beta1,
        beta2,
        weight_decay,
        eps,
        asmgrad,
        maximize,
    )
    return success


@_optim_adamw_opt.register(
    "Function",
    "Float",
    "Float",
    "Float",
    "Tensor",
    "Bool",
    "Bool",
    "Tensor",
    "Tensor",
    "Tensor",
    "Tensor",
    "Tensor",
    "Tensor",
    "Tensor",
)
def _run_optim_adamw_opt(
    opt,
    beta1,
    beta2,
    eps,
    step,
    asmgrad,
    maximize,
    learning_rate,
    weight_decay,
    parameters,
    grads,
    exp_avg,
    exp_avg_sq,
    max_exp_avg_sq,
):
    success = True
    learning_rate = float(learning_rate)
    weight_decay = float(weight_decay)
    opt(
        parameters,
        exp_avg,
        exp_avg_sq,
        max_exp_avg_sq,
        grads.astype(parameters.dtype),
        step,
        learning_rate,
        beta1,
        beta2,
        weight_decay,
        eps,
        asmgrad,
        maximize,
    )
    return success


class AdamW(Optimizer):
    def __init__(
        self,
        params,
        learning_rate=1e-3,
        beta1=0.9,
        beta2=0.999,
        eps=1e-6,
        weight_decay=0.0,
        amsgrad=False,
        *,
        maximize=False
    ):
        super().__init__(learning_rate, params, weight_decay)
        self._check_param_value(beta1, beta2, eps, self.cls_name)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.exp_avg = self.parameters.clone(prefix="exp_avg", init="zeros")
        self.exp_avg_sq = self.parameters.clone(prefix="exp_avg_sq", init="zeros")
        if amsgrad:
            self.max_exp_avg_sq = self.parameters.clone(prefix="max_exp_avg_sq", init="zeros")
        self.adamw_opt = gen.AdamWeightDecayExt()
        self.amsgrad = amsgrad
        self.maximize = maximize

    def _check_param_value(self, beta1, beta2, eps, prim_name):
        validator.check_value_type("beta1", beta1, [float], prim_name)
        validator.check_value_type("beta2", beta2, [float], prim_name)
        validator.check_value_type("eps", eps, [float], prim_name)
        validator.check_float_range(beta1, 0.0, 1.0, validator.INC_NEITHER, "beta1", prim_name)
        validator.check_float_range(beta2, 0.0, 1.0, validator.INC_NEITHER, "beta2", prim_name)
        validator.check_positive_float(eps, "eps", prim_name)

    def construct(self, gradients):
        gradients = self.flatten_gradients(gradients)
        weight_decay = self.get_weight_decay()
        lr = self.get_lr()
        self.assignadd(self.global_step, self.global_step_increase_tensor)
        state_step = self.global_step.astype(ms.float32)
        if self.amsgrad:
            if self.is_group:
                if self.is_group_lr:
                    optim_result = self.hyper_map(
                        ops.partial(
                            _optim_adamw_opt,
                            self.adamw_opt,
                            self.beta1,
                            self.beta2,
                            self.eps,
                            state_step,
                            self.amsgrad,
                            self.maximize,
                        ),
                        lr,
                        weight_decay,
                        self._parameters,
                        gradients,
                        self.exp_avg,
                        self.exp_avg_sq,
                        self.max_exp_avg_sq,
                    )
                else:
                    optim_result = self.hyper_map(
                        ops.partial(
                            _optim_adamw_opt,
                            self.adamw_opt,
                            self.beta1,
                            self.beta2,
                            self.eps,
                            state_step,
                            self.amsgrad,
                            self.maximize,
                            lr,
                        ),
                        weight_decay,
                        self._parameters,
                        gradients,
                        self.exp_avg,
                        self.exp_avg_sq,
                        self.max_exp_avg_sq,
                    )
            else:
                optim_result = self.hyper_map(
                    ops.partial(
                        _optim_adamw_opt,
                        self.adamw_opt,
                        self.beta1,
                        self.beta2,
                        self.eps,
                        state_step,
                        self.amsgrad,
                        self.maximize,
                        lr,
                        weight_decay,
                    ),
                    self._parameters,
                    gradients,
                    self.exp_avg,
                    self.exp_avg_sq,
                    self.max_exp_avg_sq,
                )
        else:
            if self.is_group:
                if self.is_group_lr:
                    optim_result = self.hyper_map(
                        ops.partial(
                            _optim_adamw_opt,
                            self.adamw_opt,
                            self.beta1,
                            self.beta2,
                            self.eps,
                            state_step,
                            self.amsgrad,
                            self.maximize,
                        ),
                        lr,
                        weight_decay,
                        self._parameters,
                        gradients,
                        self.exp_avg,
                        self.exp_avg_sq,
                        self.max_exp_avg_sq,
                    )
                else:
                    optim_result = self.hyper_map(
                        ops.partial(
                            _optim_adamw_opt,
                            self.adamw_opt,
                            self.beta1,
                            self.beta2,
                            self.eps,
                            state_step,
                            self.amsgrad,
                            self.maximize,
                            lr,
                        ),
                        weight_decay,
                        self._parameters,
                        gradients,
                        self.exp_avg,
                        self.exp_avg_sq,
                        self.max_exp_avg_sq,
                    )
            else:
                optim_result = self.hyper_map(
                    ops.partial(
                        _optim_adamw_opt,
                        self.adamw_opt,
                        self.beta1,
                        self.beta2,
                        self.eps,
                        state_step,
                        self.amsgrad,
                        self.maximize,
                        lr,
                        weight_decay,
                    ),
                    self._parameters,
                    gradients,
                    self.exp_avg,
                    self.exp_avg_sq,
                    self.max_exp_avg_sq,
                )
        return optim_result
