from model import Model
from dataset import create_df_from_dataset
import numpy as np
import os
import librosa
import av

def read_video_pyav(container):
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
    for frame in container.decode(video=0):
        frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])



if __name__ == '__main__':
    sample_rate = 16000
    num_frames = 120
    num_hidden_layers = 4
    num_attention_heads = 4
    intermediate_size = 2000

    model = Model(
        num_frames=num_frames,
        sample_rate=sample_rate,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
    )

    dataset = create_df_from_dataset()

    for _, row in dataset.iterrows():
        audio, _ = librosa.load(row['audio'], sr=sample_rate)
        
        container = av.open(row['video'])
        video = read_video_pyav(container)

        output = model(video, audio)

        print(output)
