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
from pathlib import Path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    return accuracy, f1



def train_loop(model: torch.nn.Module, train_set, val_set, epochs, batch_size=1):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    model.train()

    accuracies = np.ones(epochs) * np.nan
    f1s = np.ones(epochs) * np.nan
    losses = np.zeros(epochs)

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        train_set = train_set.shuffle()
        batched_train = train_set.batch(batch_size=batch_size)
        for batch in tqdm(batched_train):
            videos, audios = batchify(batch)

            output = model(videos, audios).squeeze(-1)


            target = torch.tensor(batch['label'], dtype=torch.float, device=DEVICE)

            optimizer.zero_grad()
            loss = criterion(output, target)
            losses[epoch] += loss.item()
            loss.backward()
            optimizer.step()
        print(f"Loss: {losses[epoch]}")
        accuracy, f1 = evaluate(model, val_set)
        accuracies[epoch] = accuracy
        f1s[epoch] = f1
    return model.state_dict(), accuracies, f1s, losses


def save_training_res(training_path, model_state_dict, res_dict):
    """Save training results to disk."""
    model_path = training_path / "model_checkpoint.pt"
    torch.save(model_state_dict, model_path)
    print(f"Model checkpoint saved at: `{model_path.absolute()}`")
    results_path = training_path / "res.tar"
    torch.save(res_dict, results_path)
    print(f"Training results saved at: `{results_path.absolute()}`")


if __name__ == '__main__':
    sample_rate = 16000
    num_frames = 120
    num_hidden_layers = 6
    num_attention_heads = 3
    intermediate_size = 300
    val_size = 0.15
    test_size = 0.15
    epochs = 20
    batch_size = 50
    seed = 42
    training_path = Path(".")

    model = Model(
        num_frames=num_frames,
        sample_rate=sample_rate,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
    )

    dataset = Dataset.from_pandas(create_df_from_dataset())#.take(batch_size) # Take a subset of N datapoints for testing purposes

    dataset = dataset.train_test_split(test_size=test_size, seed=seed, shuffle=True)

    train_dataset = dataset['train']
    # also split for val set
    train_dataset = train_dataset.train_test_split(
        test_size=(val_size / (1 - test_size)),
        seed=seed,
        shuffle=True
    )
    

    val_dataset = train_dataset['test']
    train_dataset = train_dataset['train']
    test_dataset = dataset['test']    

    model_state_dict, accuracies, f1s, losses = train_loop(
        model=model,
        train_set=train_dataset,
        val_set=val_dataset,
        epochs=epochs,
        batch_size=batch_size
    )

    res_dict = {
        "accuracies": accuracies,
        "f1s": f1s,
        "losses": losses,
        "num_frames": num_frames,
        "sample_rate": sample_rate,
        "num_hidden_layers": num_hidden_layers,
        "num_attention_heads": num_attention_heads,
        "intermediate_size": intermediate_size,       
        "epochs": epochs,
        "seed": seed,
        "test_size": test_size,
        "val_size": val_size,
    }
    save_training_res(
        training_path=training_path,
        model_state_dict=model_state_dict,
        res_dict=res_dict
    )

    print("Test results:")
    evaluate(model, test_dataset)
