from model import Model
import numpy as np
import librosa
import os
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


sample_rate = 16000
num_frames = 120

audio_file_path = os.getcwd() + "/datasets/4_second_base_dataset_misaligned/Mazda3_processed_audio4seconds/Mazda3_30_offset-2.wav"
video_file_path = os.getcwd() + "/datasets/4_second_base_dataset_misaligned/Mazda3_processed_video4seconds/Mazda3_30.MP4"

container = av.open(video_file_path)

video = read_video_pyav(container=container)

audio, sample_rate = librosa.load(audio_file_path, sr=sample_rate)

model = Model(num_frames=num_frames, sample_rate=sample_rate)

output = model(video, audio)

print(output)