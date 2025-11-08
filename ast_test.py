import librosa
import numpy as np
import os

from transformers import AutoFeatureExtractor, ASTConfig, ASTModel

MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"

# Load the audio file

file_path = os.getcwd() + "/datasets/base_dataset/Mazda3_base/Mazda3_70.wav"

sample_rate = 16000

# sr=None preserves the original sample rate, otherwise it resamples to 22050 Hz by default
audio, sample_rate = librosa.load(file_path, sr=sample_rate)

feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)

config = ASTConfig(
            hidden_size=300
        )

model = ASTModel(config)

inputs = feature_extractor(audio, sampling_rate=sample_rate, return_tensors='pt') 

outputs = model(**inputs, output_hidden_states=True)

last_hidden_states = outputs.last_hidden_state

print(last_hidden_states.shape)