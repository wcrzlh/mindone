from typing import List, Union

VideoInput = Union[
    List["PIL.Image.Image"],
    "np.ndarray",
    "ms.Tensor",
    List["np.ndarray"],
    List["ms.Tensor"],
    List[List["PIL.Image.Image"]],
    List[List["np.ndarrray"]],
    List[List["ms.Tensor"]],
]  # noqa