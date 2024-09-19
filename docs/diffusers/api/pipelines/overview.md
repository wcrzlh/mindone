<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Pipelines

Pipelines provide a simple way to run state-of-the-art diffusion models in inference by bundling all of the necessary components (multiple independently-trained models, schedulers, and processors) into a single end-to-end class. Pipelines are flexible and they can be adapted to use different schedulers or even model components.

All pipelines are built from the base [`DiffusionPipeline`] class which provides basic functionality for loading, downloading, and saving all the components. Specific pipeline types (for example [`StableDiffusionPipeline`]) loaded with [`~DiffusionPipeline.from_pretrained`] are automatically detected and the pipeline components are loaded and passed to the `__init__` function of the pipeline.

<Tip warning={true}>

You shouldn't use the [`DiffusionPipeline`] class for training. Individual components (for example, [`UNet2DModel`] and [`UNet2DConditionModel`]) of diffusion pipelines are usually trained individually, so we suggest directly working with them instead.

<br>

Pipelines do not offer any training functionality. You'll notice MindSpore's autograd is disabled by decorating the [`~DiffusionPipeline.__call__`] method with a [`mindspore._no_grad`] decorator because pipelines should not be used for training. If you're interested in training, please take a look at the [Training](../../training/overview) guides instead!

</Tip>

The table below lists all the pipelines currently available in 🤗 Diffusers and the tasks they support. Click on a pipeline to view its abstract and published paper.

| Pipeline                                                       | Tasks |
|----------------------------------------------------------------|---|
| [AnimateDiff](../animatediff)                                  | text2video |
| [BLIP Diffusion](../blip_diffusion)                            | text2image |
| [Consistency Models](../consistency_models)                    | unconditional image generation |
| [ControlNet](../controlnet)                                    | text2image, image2image, inpainting |
| [ControlNet with Stable Diffusion 3](../controlnet_sd3)        | text2image |
| [ControlNet with Stable Diffusion XL](../controlnet_sdxl)      | text2image |
| [ControlNet-XS](../controlnetxs)                               | text2image |
| [ControlNet-XS with Stable Diffusion XL](../controlnetxs_sdxl) | text2image |
| [Dance Diffusion](../dance_diffusion)                          | unconditional audio generation |
| [DDIM](../ddim)                                                | unconditional image generation |
| [DDPM](../ddpm)                                                | unconditional image generation |
| [DeepFloyd IF](../deepfloyd_if)                                | text2image, image2image, inpainting, super-resolution |
| [DiffEdit](../diffedit)                                        | inpainting |
| [DiT](../dit)                                                  | text2image |
| [Hunyuan-DiT](../hunyuandit)                                   | text2image |
| [I2VGen-XL](../i2vgenxl)                                       | text2video |
| [InstructPix2Pix](../pix2pix)                                  | image editing |
| [Kandinsky 2.1](../kandinsky)                                  | text2image, image2image, inpainting, interpolation |
| [Kandinsky 2.2](../kandinsky_v22)                              | text2image, image2image, inpainting |
| [Kandinsky 3](../kandinsky3)                                   | text2image, image2image |
| [Latent Consistency Models](../latent_consistency_models)      | text2image |
| [Latent Diffusion](../latent_diffusion)                        | text2image, super-resolution |
| [Marigold](../marigold)                                        | depth |
| [PixArt-α](../pixart)                                          | text2image |
| [PixArt-Σ](../pixart_sigma)                                    | text2image |
| [Shap-E](../shap_e)                                            | text-to-3D, image-to-3D |
| [Stable Cascade](../stable_cascade)                            | text2image |
| [unCLIP](../unclip)                                            | text2image, image variation |
| [Wuerstchen](../wuerstchen)                                    | text2image |


::: mindone.diffusers.DiffusionPipeline
	members:
		- all
		- __call__
		- device
		- to
		- components

::: mindone.diffusers.utils.PushToHubMixin
