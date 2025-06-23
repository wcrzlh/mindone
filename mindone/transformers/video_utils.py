# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from collections.abc import Iterable
from contextlib import redirect_stdout
from dataclasses import dataclass
from io import BytesIO
from typing import Callable, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse

import numpy as np
import requests

from .image_transforms import PaddingMode, to_channel_dimension_format
from .image_utils import ChannelDimension, infer_channel_dimension_format, is_valid_image
from transformers.utils import (
    is_av_available,
    is_cv2_available,
    is_decord_available,
    is_numpy_array,
    is_vision_available,
    is_yt_dlp_available,
    logging,
    requires_backends,
)
from .utils import (
    is_numpy_array,
    is_mindspore_available,
    is_mindspore_tensor,
)


if is_vision_available():
    import PIL.Image
    import PIL.ImageOps

if is_mindspore_available():
    import mindspore as ms
    from mindspore import mint, ops


logger = logging.get_logger(__name__)


VideoInput = Union[
    List["PIL.Image.Image"],
    "np.ndarray",
    "torch.Tensor",
    List["np.ndarray"],
    List["torch.Tensor"],
    List[List["PIL.Image.Image"]],
    List[List["np.ndarrray"]],
    List[List["torch.Tensor"]],
]  # noqa

@dataclass
class VideoMetadata:
    total_num_frames: int
    fps: float
    duration: float
    video_backend: str

    def __getitem__(self, item):
        return getattr(self, item)

def is_valid_video_frame(frame):
    return isinstance(frame, PIL.Image.Image) or (
        (is_numpy_array(frame) or is_mindspore_tensor(frame)) and frame.ndim == 3
    )


def is_valid_video(video):
    if not isinstance(video, (list, tuple)):
        return (is_numpy_array(video) or is_mindspore_tensor(video)) and video.ndim == 4
    return all(is_valid_video_frame(frame) for frame in video)


def valid_videos(videos):
    # If we have a list of videos, it could be either one video as list of frames or a batch
    if isinstance(videos, (list, tuple)):
        for video_or_frame in videos:
            if not (is_valid_video(video_or_frame) or is_valid_video_frame(video_or_frame)):
                return False
    # If not a list, then we have a single 4D video or 5D batched tensor
    elif not is_valid_video(videos) or videos.ndim == 5:
        return False
    return True


def is_batched_video(videos):
    if isinstance(videos, (list, tuple)):
        return is_valid_video(videos[0])
    elif (is_numpy_array(videos) or is_mindspore_tensor(videos)) and videos.ndim == 5:
        return True
    return False


def is_scaled_video(video: np.ndarray) -> bool:
    """
    Checks to see whether the pixel values have already been rescaled to [0, 1].
    """
    # It's possible the video has pixel values in [0, 255] but is of floating type
    return np.min(video) >= 0 and np.max(video) <= 1


def convert_pil_frames_to_video(videos: List[VideoInput]) -> List[Union["np.ndarray", "torch.Tensor"]]:
    """
    Given a batch of videos, converts each video to a 4D array. If video is already in array type,
    it is simply returned. We assume that all inputs in the list are in the same format, based on the type of the first element.

    Args:
        videos (`VideoInput`):
            Video inputs to turn into a list of videos.
    """

    if not isinstance(videos[0], (list, tuple)):
        return videos

    video_converted = []
    for video in videos:
        video = [np.array(frame) for frame in video]
        video = np.stack(video)
        video_converted.append(video)
    return video_converted
def make_batched_videos(videos) -> List[Union["np.ndarray", "torch.Tensor"]]:
    """
    Ensure that the input is a list of videos. If the input is a single video, it is converted to a list of length 1.
    If the input is a batch of videos, it is converted to a list of 4D video arrays. Videos passed as list `PIL.Image`
    frames are converted to 4D arrays.

    We assume that all inputs in the list are in the same format, based on the type of the first element.

    Args:
        videos (`VideoInput`):
            Video inputs to turn into a list of videos.
    """
    if not valid_videos:
        raise ValueError(
            f"Invalid video input. Expected either a list of video frames or an input of 4 or 5 dimensions, but got"
            f" type {type(videos)}."
        )

    if is_batched_video(videos):
        pass
    elif is_valid_video(videos):
        videos = [videos]
    # only one frame passed, thus we unsqueeze time dim
    elif is_valid_image(videos):
        videos = [np.array(videos)[None, ...]]
    # nested batch so we need to unflatten
    elif isinstance(videos[0], (list, tuple)) and is_valid_video(videos[0][0]):
        videos = [video for sublist in videos for video in sublist]
    return convert_pil_frames_to_video(videos)

def default_sample_indices_fn(metadata: VideoMetadata, num_frames=None, fps=None, **kwargs):
    """
    A default sampling function that replicates the logic used in get_uniform_frame_indices,
    while optionally handling `fps` if `num_frames` is not provided.

    Args:
        metadata (`VideoMetadata`):
            `VideoMetadata` object containing metadata about the video, such as "total_num_frames" or "fps".
        num_frames (`int`, *optional*):
            Number of frames to sample uniformly.
        fps (`int`, *optional*):
            Desired frames per second. Takes priority over num_frames if both are provided.

    Returns:
        `np.ndarray`: Array of frame indices to sample.
    """
    total_num_frames = metadata.total_num_frames
    video_fps = metadata.fps

    # If num_frames is not given but fps is, calculate num_frames from fps
    if num_frames is None and fps is not None:
        num_frames = int(total_num_frames / video_fps * fps)
        if num_frames > total_num_frames:
            raise ValueError(
                f"When loading the video with fps={fps}, we computed num_frames={num_frames} "
                f"which exceeds total_num_frames={total_num_frames}. Check fps or video metadata."
            )

    if num_frames is not None:
        indices = np.arange(0, total_num_frames, total_num_frames / num_frames, dtype=int)
    else:
        indices = np.arange(0, total_num_frames, dtype=int)
    return indices

@dataclass
class VideoMetadata:
    total_num_frames: int
    fps: float
    duration: float
    video_backend: str

    def __getitem__(self, item):
        return getattr(self, item)

def read_video_opencv(
    video_path: str,
    sample_indices_fn: Callable,
    **kwargs,
):
    """
    Decode a video using the OpenCV backend.

    Args:
        video_path (`str`):
            Path to the video file.
        sample_indices_fn (`Callable`):
            A callable function that will return indices at which the video should be sampled. If the video has to be loaded using
            by a different sampling technique than provided by `num_frames` or `fps` arguments, one should provide their own `sample_indices_fn`.
            If not provided, simple uniform sampling with fps is performed.
            Example:
            def sample_indices_fn(metadata, **kwargs):
                return np.linspace(0, metadata.total_num_frames - 1, num_frames, dtype=int)

    Returns:
        Tuple[`np.array`, `VideoMetadata`]: A tuple containing:
            - Numpy array of frames in RGB (shape: [num_frames, height, width, 3]).
            - `VideoMetadata` object.
    """
    # Lazy import cv2
    requires_backends(read_video_opencv, ["cv2"])
    import cv2

    video = cv2.VideoCapture(video_path)
    total_num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = video.get(cv2.CAP_PROP_FPS)
    duration = total_num_frames / video_fps if video_fps else 0
    metadata = VideoMetadata(
        total_num_frames=int(total_num_frames), fps=float(video_fps), duration=float(duration), video_backend="opencv"
    )
    indices = sample_indices_fn(metadata=metadata, **kwargs)

    index = 0
    frames = []
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        if index in indices:
            height, width, channel = frame.shape
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame[0:height, 0:width, 0:channel])
        if success:
            index += 1
        if index >= total_num_frames:
            break

    video.release()
    metadata.frames_indices = indices
    return np.stack(frames), metadata
def read_video_decord(
    video_path: str,
    sample_indices_fn: Optional[Callable] = None,
    **kwargs,
):
    """
    Decode a video using the Decord backend.

    Args:
        video_path (`str`):
            Path to the video file.
        sample_indices_fn (`Callable`, *optional*):
            A callable function that will return indices at which the video should be sampled. If the video has to be loaded using
            by a different sampling technique than provided by `num_frames` or `fps` arguments, one should provide their own `sample_indices_fn`.
            If not provided, simple uniform sampling with fps is performed.
            Example:
            def sample_indices_fn(metadata, **kwargs):
                return np.linspace(0, metadata.total_num_frames - 1, num_frames, dtype=int)

    Returns:
        Tuple[`np.array`, `VideoMetadata`]: A tuple containing:
            - Numpy array of frames in RGB (shape: [num_frames, height, width, 3]).
            - `VideoMetadata` object.
    """
    # Lazy import from decord
    requires_backends(read_video_decord, ["decord"])
    from decord import VideoReader, cpu

    vr = VideoReader(uri=video_path, ctx=cpu(0))  # decord has problems with gpu
    video_fps = vr.get_avg_fps()
    total_num_frames = len(vr)
    duration = total_num_frames / video_fps if video_fps else 0
    metadata = VideoMetadata(
        total_num_frames=int(total_num_frames), fps=float(video_fps), duration=float(duration), video_backend="decord"
    )

    indices = sample_indices_fn(metadata=metadata, **kwargs)

    frames = vr.get_batch(indices).asnumpy()
    metadata.frames_indices = indices
    return frames, metadata


def read_video_pyav(
    video_path: str,
    sample_indices_fn: Callable,
    **kwargs,
):
    """
    Decode the video with PyAV decoder.

    Args:
        video_path (`str`):
            Path to the video file.
        sample_indices_fn (`Callable`, *optional*):
            A callable function that will return indices at which the video should be sampled. If the video has to be loaded using
            by a different sampling technique than provided by `num_frames` or `fps` arguments, one should provide their own `sample_indices_fn`.
            If not provided, simple uniform sampling with fps is performed.
            Example:
            def sample_indices_fn(metadata, **kwargs):
                return np.linspace(0, metadata.total_num_frames - 1, num_frames, dtype=int)

    Returns:
        Tuple[`np.array`, `VideoMetadata`]: A tuple containing:
            - Numpy array of frames in RGB (shape: [num_frames, height, width, 3]).
            - `VideoMetadata` object.
    """
    # Lazy import av
    requires_backends(read_video_pyav, ["av"])
    import av

    container = av.open(video_path)
    total_num_frames = container.streams.video[0].frames
    video_fps = container.streams.video[0].average_rate  # should we better use `av_guess_frame_rate`?
    duration = total_num_frames / video_fps if video_fps else 0
    metadata = VideoMetadata(
        total_num_frames=int(total_num_frames), fps=float(video_fps), duration=float(duration), video_backend="pyav"
    )
    indices = sample_indices_fn(metadata=metadata, **kwargs)

    frames = []
    container.seek(0)
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= 0 and i in indices:
            frames.append(frame)

    video = np.stack([x.to_ndarray(format="rgb24") for x in frames])
    metadata.frames_indices = indices
    return video, metadata

VIDEO_DECODERS = {
    "decord": read_video_decord,
    "opencv": read_video_opencv,
    "pyav": read_video_pyav,
}

def load_video(
    video: Union[str, "VideoInput"],
    num_frames: Optional[int] = None,
    fps: Optional[int] = None,
    backend: str = "pyav",
    sample_indices_fn: Optional[Callable] = None,
    **kwargs,
) -> np.array:
    """
    Loads `video` to a numpy array.

    Args:
        video (`str` or `VideoInput`):
            The video to convert to the numpy array format. Can be a link to video or local path.
        num_frames (`int`, *optional*):
            Number of frames to sample uniformly. If not passed, the whole video is loaded.
        fps (`int`, *optional*):
            Number of frames to sample per second. Should be passed only when `num_frames=None`.
            If not specified and `num_frames==None`, all frames are sampled.
        backend (`str`, *optional*, defaults to `"pyav"`):
            The backend to use when loading the video. Can be any of ["decord", "pyav", "opencv", "torchvision"]. Defaults to "pyav".
        sample_indices_fn (`Callable`, *optional*):
            A callable function that will return indices at which the video should be sampled. If the video has to be loaded using
            by a different sampling technique than provided by `num_frames` or `fps` arguments, one should provide their own `sample_indices_fn`.
            If not provided, simple uniformt sampling with fps is performed, otherwise `sample_indices_fn` has priority over other args.
            The function expects at input the all args along with all kwargs passed to `load_video` and should output valid
            indices at which the video should be sampled. For example:

            Example:
            def sample_indices_fn(metadata, **kwargs):
                return np.linspace(0, metadata.total_num_frames - 1, num_frames, dtype=int)

    Returns:
        Tuple[`np.array`, Dict]: A tuple containing:
            - Numpy array of frames in RGB (shape: [num_frames, height, width, 3]).
            - Metadata dictionary.
    """

    # If `sample_indices_fn` is given, we can accept any args as those might be needed by custom `sample_indices_fn`
    if fps is not None and num_frames is not None and sample_indices_fn is None:
        raise ValueError(
            "`num_frames`, `fps`, and `sample_indices_fn` are mutually exclusive arguments, please use only one!"
        )

    # If user didn't pass a sampling function, create one on the fly with default logic
    if sample_indices_fn is None:

        def sample_indices_fn_func(metadata, **fn_kwargs):
            return default_sample_indices_fn(metadata, num_frames=num_frames, fps=fps, **fn_kwargs)

        sample_indices_fn = sample_indices_fn_func

    if urlparse(video).netloc in ["www.youtube.com", "youtube.com"]:
        if not is_yt_dlp_available():
            raise ImportError("To load a video from YouTube url you have  to install `yt_dlp` first.")
        else:
            # Lazy import from yt_dlp
            requires_backends(load_video, ["yt_dlp"])
            from yt_dlp import YoutubeDL

        buffer = BytesIO()
        with redirect_stdout(buffer), YoutubeDL() as f:
            f.download([video])
        bytes_obj = buffer.getvalue()
        file_obj = BytesIO(bytes_obj)
    elif video.startswith("http://") or video.startswith("https://"):
        file_obj = BytesIO(requests.get(video).content)
    elif os.path.isfile(video):
        file_obj = video
    elif is_valid_image(video) or (isinstance(video, (list, tuple)) and is_valid_image(video[0])):
        file_obj = None
    else:
        raise TypeError("Incorrect format used for video. Should be an url linking to an video or a local path.")

    # can also load with decord, but not cv2/torchvision
    # both will fail in case of url links
    video_is_url = video.startswith("http://") or video.startswith("https://")
    if video_is_url and backend in ["opencv", "torchvision"]:
        raise ValueError(
            "If you are trying to load a video from URL, you can decode the video only with `pyav` or `decord` as backend"
        )

    if file_obj is None:
        return video

    if (
        (not is_decord_available() and backend == "decord")
        or (not is_av_available() and backend == "pyav")
        or (not is_cv2_available() and backend == "opencv")
    ):
        raise ImportError(
            f"You chose backend={backend} for loading the video but the required library is not found in your environment "
            f"Make sure to install {backend} before loading the video."
        )

    video_decoder = VIDEO_DECODERS[backend]
    video, metadata = video_decoder(file_obj, sample_indices_fn, **kwargs)
    return video, metadata


def convert_to_rgb(
    video: np.array,
    data_format: Optional[ChannelDimension] = None,
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
) -> np.array:
    """
    Convert video to RGB by blending the transparency layer if it's in RGBA format, otherwise simply returns it.

    Args:
        video (`np.array`):
            The video to convert.
        data_format (`ChannelDimension`, *optional*):
            The channel dimension format of the output video. If unset, will use the inferred format from the input.
        input_data_format (`ChannelDimension`, *optional*):
            The channel dimension format of the input video. If unset, will use the inferred format from the input.
    """
    if not isinstance(video, np.ndarray):
        raise ValueError(f"Video has to be a numpy array to convert to RGB format, but found {type(video)}")

    # np.array usually comes with ChannelDimension.LAST so leet's convert it
    if input_data_format is None:
        input_data_format = infer_channel_dimension_format(video)
    video = to_channel_dimension_format(video, ChannelDimension.FIRST, input_channel_dim=input_data_format)

    # 3 channels for RGB already
    if video.shape[-3] == 3:
        return video

    # Grayscale video so we repeat it 3 times for each channel
    if video.shape[-3] == 1:
        return video.repeat(3, -3)

    if not (video[..., 3, :, :] < 255).any():
        return video

    # There is a transparency layer, blend it with a white background.
    # Calculate the alpha proportion for blending.
    alpha = video[..., 3, :, :] / 255.0
    video = (1 - alpha[..., None, :, :]) * 255 + alpha[..., None, :, :] * video[..., 3, :, :]
    return video


def group_videos_by_shape(
    videos: List["torch.Tensor"],
) -> Tuple[Dict[Tuple[int, int], List["torch.Tensor"]], Dict[int, Tuple[Tuple[int, int], int]]]:
    """
    Groups videos by shape.
    Returns a dictionary with the shape as key and a list of videos with that shape as value,
    and a dictionary with the index of the video in the original list as key and the shape and index in the grouped list as value.
    """
    grouped_videos = {}
    grouped_videos_index = {}
    for i, video in enumerate(videos):
        shape = video.shape[-2::]
        num_frames = video.shape[-4]  # video format BTCHW
        shape = (num_frames, *shape)
        if shape not in grouped_videos:
            grouped_videos[shape] = []
        grouped_videos[shape].append(video)
        grouped_videos_index[i] = (shape, len(grouped_videos[shape]) - 1)
    # stack videos with the same size and number of frames
    grouped_videos = {shape: mint.stack(videos, dim=0) for shape, videos in grouped_videos.items()}
    return grouped_videos, grouped_videos_index


def reorder_videos(
    processed_videos: Dict[Tuple[int, int], "torch.Tensor"], grouped_videos_index: Dict[int, Tuple[int, int]]
) -> List["torch.Tensor"]:
    """
    Reconstructs a list of videos in the original order.
    """
    return [
        processed_videos[grouped_videos_index[i][0]][grouped_videos_index[i][1]]
        for i in range(len(grouped_videos_index))
    ]
