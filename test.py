from model import Model
from datasets import Dataset
from dataset import create_df_from_dataset
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
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

def batchify(batch):
    videos, audios = [], []
    for i in range(len(batch['audio'])):
        audio, _ = librosa.load(batch['audio'][i], sr=sample_rate)
        audios.append(audio)

        container = av.open(batch['video'][i])
        video = read_video_pyav(container)
        videos.append(video)
    
    return videos, audios



def train_loop(model: torch.nn.Module, train_set, val_set, epochs, batch_size=2):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters())

    percent_epoch = int(len(train_set)/2) # report stats every half epoch

    for epoch in range(epochs):
        print(f"epoch {epoch}:")
        train_set = train_set.shuffle()
        batched_train = train_set.batch(batch_size=2)
        for i, batch in enumerate(batched_train):
            if (i + 1) % percent_epoch == 0:
                print("half epoch!")

            videos, audios = batchify(batch)

            output = model(videos, audios).squeeze(-1)

            target = torch.tensor(batch['label'], dtype=torch.float)

            model.zero_grad()
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            exit(0)


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

    VAL_SIZE = 0.15
    TEST_SIZE = 0.15

    dataset = Dataset.from_pandas(create_df_from_dataset())

    dataset = dataset.train_test_split(test_size=TEST_SIZE)

    train_dataset = dataset['train']
    # also split for val set
    train_dataset = train_dataset.train_test_split(test_size=(VAL_SIZE / (1 - TEST_SIZE)))
    

    val_dataset = train_dataset['test']
    train_dataset = train_dataset['train']
    test_dataset = dataset['test']    

    train_loop(model, train_set=train_dataset, val_set=val_dataset, epochs=50)