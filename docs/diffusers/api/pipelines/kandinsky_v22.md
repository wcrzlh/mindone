<!--Copyright 2024 The HuggingFace Team. All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Kandinsky 2.2

Kandinsky 2.2 is created by [Arseniy Shakhmatov](https://github.com/cene555), [Anton Razzhigaev](https://github.com/razzant), [Aleksandr Nikolich](https://github.com/AlexWortega), [Vladimir Arkhipkin](https://github.com/oriBetelgeuse), [Igor Pavlov](https://github.com/boomb0om), [Andrey Kuznetsov](https://github.com/kuznetsoffandrey), and [Denis Dimitrov](https://github.com/denndimitrov).

The description from it's GitHub page is:

*Kandinsky 2.2 brings substantial improvements upon its predecessor, Kandinsky 2.1, by introducing a new, more powerful image encoder - CLIP-ViT-G and the ControlNet support. The switch to CLIP-ViT-G as the image encoder significantly increases the model's capability to generate more aesthetic pictures and better understand text, thus enhancing the model's overall performance. The addition of the ControlNet mechanism allows the model to effectively control the process of generating images. This leads to more accurate and visually appealing outputs and opens new possibilities for text-guided image manipulation.*

The original codebase can be found at [ai-forever/Kandinsky-2](https://github.com/ai-forever/Kandinsky-2).

<Tip>

Check out the [Kandinsky Community](https://huggingface.co/kandinsky-community) organization on the Hub for the official model checkpoints for tasks like text-to-image, image-to-image, and inpainting.

</Tip>

<Tip>

Make sure to check out the schedulers [guide](../../using-diffusers/schedulers) to learn how to explore the tradeoff between scheduler speed and quality, and see the [reuse components across pipelines](../../using-diffusers/loading#reuse-components-across-pipelines) section to learn how to efficiently load the same components into multiple pipelines.

</Tip>

::: mindone.diffusers.KandinskyV22PriorPipeline
    members:
        - all
        - __call__
        - interpolate

::: mindone.diffusers.KandinskyV22Pipeline
    members:
        - all
        - __call__

::: mindone.diffusers.KandinskyV22CombinedPipeline
    members:
        - all
        - __call__

::: mindone.diffusers.KandinskyV22ControlnetPipeline
    members:
        - all
        - __call__

::: mindone.diffusers.KandinskyV22PriorEmb2EmbPipeline
    members:
        - all
        - __call__
        - interpolate

::: mindone.diffusers.KandinskyV22Img2ImgPipeline
    members:
        - all
        - __call__

::: mindone.diffusers.KandinskyV22Img2ImgCombinedPipeline
    members:
        - all
        - __call__

::: mindone.diffusers.KandinskyV22ControlnetImg2ImgPipeline
    members:
        - all
        - __call__

::: mindone.diffusers.KandinskyV22InpaintPipeline
    members:
        - all
        - __call__

::: mindone.diffusers.KandinskyV22InpaintCombinedPipeline
    members:
        - all
        - __call__
