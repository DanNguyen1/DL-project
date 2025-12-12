#!/usr/bin/env python3

import av
import librosa
import numpy as np


def read_video_pyav(container):
    """
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`list[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    """
    frames = []
    container.seek(0)
    for frame in container.decode(video=0):
        frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def batchify(batch, sample_rate):
    videos, audios = [], []
    for i in range(len(batch["audio"])):
        audio, _ = librosa.load(batch["audio"][i], sr=sample_rate)
        audios.append(audio)

        container = av.open(batch["video"][i])
        video = read_video_pyav(container)
        videos.append(video)

    return videos, audios
