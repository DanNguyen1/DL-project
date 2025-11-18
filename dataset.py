import os
import re
import numpy as np
import pandas as pd

AUDIO_DIR = os.getcwd() + "/datasets/processed_dataset_4secs/processed_audio_4secs"
VIDEO_DIR = os.getcwd() + "/datasets/processed_dataset_4secs/processed_video_4secs"

def create_df_from_dataset():
    audio_files = os.listdir(AUDIO_DIR)
    video_files = os.listdir(VIDEO_DIR)

    name_mapping = {}

    for video_file in video_files:
        name_mapping[video_file[:-4]] = {"video": video_file, "audio": []}

    audio_file_root_name_pattern = r".+_\d+" 
    for audio_file in audio_files:
        match = re.search(audio_file_root_name_pattern, audio_file)
        if not match:
            continue

        root_name = match.group()

        name_mapping[root_name]["audio"].append(audio_file)
    
    video = []
    audio = []
    label = []

    aligned_audio_pattern = r".+_\d+_offset\+0\.wav"
    
    for i, video_audio_map in enumerate(name_mapping.values()):
        full_video_path = VIDEO_DIR + "/" + video_audio_map["video"]
        # encode video in numpy format

        for current_audio in video_audio_map["audio"]:
            current_label = bool(re.search(aligned_audio_pattern, current_audio))
            full_audio_path = AUDIO_DIR + "/" + current_audio
            # # encode audio in numpy format

            # put everything together
            video.append(full_video_path)
            audio.append(full_audio_path)
            label.append(current_label)

    data = {'video': video,
            'audio': audio,
            'label': label}
    
    df = pd.DataFrame(data)

    return df

