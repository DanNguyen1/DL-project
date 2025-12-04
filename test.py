from model import Model
from datasets import Dataset
from dataset import create_df_from_dataset
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import librosa
import av
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from tqdm import tqdm

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

@torch.no_grad()
def evaluate(model: torch.nn.Module, data):
    model.eval()
    videos, audios = batchify(data)

    output = torch.round(torch.sigmoid(model(videos, audios)).squeeze(-1)).numpy()

    target = np.array(data['label'])

    accuracy = accuracy_score(target, output)
    precision = precision_score(target, output)
    recall = recall_score(target, output)
    f1 = f1_score(target, output)

    print(f"accuracy: {accuracy}, precision: {precision}, recall: {recall}, f1: {f1}")



def train_loop(model: torch.nn.Module, train_set, val_set, epochs, batch_size=1):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        print(f"epoch {epoch}:")
        model.train()
        train_set = train_set.shuffle()
        batched_train = train_set.batch(batch_size=batch_size)
        for batch in tqdm(batched_train):
            videos, audios = batchify(batch)

            output = model(videos, audios).squeeze(-1)


            target = torch.tensor(batch['label'], dtype=torch.float)

            optimizer.zero_grad()
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        evaluate(model, val_set)



if __name__ == '__main__':
    sample_rate = 16000
    num_frames = 120
    num_hidden_layers = 6
    num_attention_heads = 3
    intermediate_size = 300

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

    train_loop(model, train_set=train_dataset, val_set=val_dataset, epochs=20, batch_size=50)

    print("test results:")
    evaluate(model, test_dataset)
