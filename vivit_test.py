import av
import numpy as np
import os

from transformers import VivitConfig, VivitImageProcessor, VivitModel
from huggingface_hub import hf_hub_download

np.random.seed(0)


def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`list[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    '''
    Sample a given number of frame indices from the video.
    Args:
        clip_len (`int`): Total number of frames to sample.
        frame_sample_rate (`int`): Sample every n-th frame.
        seg_len (`int`): Maximum allowed index of sample's last frame.
    Returns:
        indices (`list[int]`): List of sampled frame indices
    '''
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices


# video clip consists of 300 frames (10 seconds at 30 FPS)
file_path = os.getcwd() + "/datasets/base_dataset/Mazda3_base/Mazda3_70.MP4"

container = av.open(file_path)

# sample 32 frames
indices = sample_frame_indices(clip_len=120, frame_sample_rate=1, seg_len=container.streams.video[0].frames)
video = read_video_pyav(container=container, indices=indices)

image_processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")

config = VivitConfig(
            hidden_size=300,
            num_frames=120
        )

model = VivitModel(config)

# prepare video for the model
inputs = image_processor(list(video), return_tensors="pt")

# forward pass
outputs = model(**inputs, output_hidden_states=True)
last_hidden_state = outputs.last_hidden_state  # (batch_size, num_patches, hidden_size)

#vidoe embed is the output of the CLS token in the last hidden state
video_embed = last_hidden_state[:, -1, :]
print(video_embed.shape)
# patch_tokens = last_hidden_states[:, 1:, :]

# num_patches_per_frame = 196
# patch_tokens = patch_tokens.view(1, 60, num_patches_per_frame, 600)
# frame_embeddings = patch_tokens.mean(dim=2)     # [1, 32, 300]
# print(frame_embeddings.shape)