import torch
import torch.nn as nn
import av
from transformers import AutoFeatureExtractor, ASTConfig, ASTModel
from transformers import VivitConfig, VivitImageProcessor, VivitModel

class Model(nn.Module):
    def __init__(self,
                 num_frames,
                 hidden_size=300,
                 video_model="google/vivit-b-16x2-kinetics400",
                 audio_model="MIT/ast-finetuned-audioset-10-10-0.4593",
                 sample_rate=16000,
                 num_hidden_layers=4,
                 num_attention_heads=4,
                 intermediate_size=2000,
                 ):
        super(Model, self).__init__()
        self.num_frames = num_frames
        self.hidden_size = hidden_size
        self.video_model = video_model
        self.audio_model = audio_model
        self.sample_rate = sample_rate
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size

        self.audio_feature_extractor = AutoFeatureExtractor.from_pretrained(self.audio_model)
        audio_config = ASTConfig(hidden_size=self.hidden_size)
        self.audio_model = ASTModel(audio_config)

        self.image_processor = VivitImageProcessor.from_pretrained(self.video_model)

        video_config = VivitConfig(
            hidden_size=self.hidden_size,
            num_frames=self.num_frames,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
        )
        self.video_model = VivitModel(video_config)

        # classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 4),
            nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, video, audio):
        # create inputs for models
        video_input = []
        audio_input = []

        for i in range(len(video)):
            video_tensor = self.image_processor(list(video[i]), return_tensors="pt")['pixel_values']
            audio_tensor = self.audio_feature_extractor(audio[i], sampling_rate=self.sample_rate, return_tensors='pt') ['input_values']

            video_input.append(video_tensor)
            audio_input.append(audio_tensor)
        
        video_input = torch.concat(video_input, dim=0)
        audio_input = torch.concat(audio_input, dim=0)


        # video embed
        video_outputs = self.video_model(video_input, output_hidden_states=True)
        video_embed = video_outputs.pooler_output

        # audio embed
        audio_outputs = self.audio_model(audio_input, output_hidden_states=True)
        audio_embed = audio_outputs.pooler_output

        # classification output layer
        video_audio_embed = torch.concat((video_embed, audio_embed), dim=1)

        return self.classifier(video_audio_embed)
