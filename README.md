# COMP.5530 Deep Learning Project

## Original Dataset
- [Paper](https://doi.org/10.48550/arXiv.2212.01651), [Download](https://slobodan.ucg.ac.me/science/vs13/)

## Our Dataset

Link to our modified dataset: https://drive.google.com/drive/folders/1Gy04FZZr4mD0OiqadZck5LINrpo88TaY?usp=sharing

## Setting up an environment

Our project was desgined to work for Python >3.10. Please note that our project may not work with any previous versions of Python.

We highly recommend setting up [Python virtual environment](https://docs.python.org/3/library/venv.html). We detail how to do so below.

1. Navigate to root directory:
```bash
cd DL-project
```

2. Set up a virtual environment (OPTIONAL, BUT RECOMMENDED!)
```bash
python3 -m venv venv/
```
3. Activate the virtual environment (OPTIONAL, BUT RECOMMENDED!)
```bash
source venv/bin/activate
```

4. Download packages
```bash
python3 -m pip install -Ue .
```

5. Download our dataset and move/extract it to the root directory
Manually download the dataset [here](https://drive.google.com/drive/folders/1Gy04FZZr4mD0OiqadZck5LINrpo88TaY?usp=sharing) or the zip file [here](https://drive.google.com/file/d/14w3N7rXSLToA3YDO5bGhGQAXWUCliaBX/view).


6. Run the code
The bulk of our project is in `training.py`. You'll want to run that, but you can run any file in the repository.
```bash
python3 training.py
```

NOTE: If you are on the CS GPU servers, you can restrict our code to run on a single GPU with the following environment variable:
```bash
CUDA_VISIBLE_DEVICES=<GPU_INDEX> python3 training.py
```
where you change <GPU_INDEX> to a certain GPU. You can check GPUs on the server by running:
```bash
nvidia-smi
```

### Files
- `dataset.py`
Utility file for parsing the files in our dataset, associating pairs of video and audio files with each other, and labelling them. Puts **ONLY** the audio and video **FILE NAMES** and labels into a Pandas dataframe for later processing.

- `ast_test.py`
Initial testing file to test the capabilities of [Audio Spectrogram Transformer](https://huggingface.co/docs/transformers/en/model_doc/audio-spectrogram-transformer). Runs AST on a sample audio file from our dataset and produces an embedding.

- `vivit_test.py`
Initial testing file to test the capabilities of [Video Vision Transformer](https://huggingface.co/docs/transformers/en/model_doc/vivit). Runs ViViT on a samplpe video file from our dataset and produces an embedding.

- `model.py`
Our main model. Sets up a torch-based module for learning video-audio alignment through pairs of video and audio files. Does not perform initial vectorization of video and audio data, but will perform AST and ViViT specific processing before passing it through each individual model. The forward method will produce video and audio embeddings through AST and ViViT, respectively, concatenate the embedding pairs, and pass the concatenated embeddings through a feedforward network to produce a binary classification.

- `training.py`
The main file to train the model. This file will run the setup for the model and the training loop to train the model on the video-audio alignment detection task. It will first perform a training, validation, and test split of the dataset. The main training loop will batch the example pairs every epoch and train the model in mini-batches. Additionally, this file will evaluate the model on accuracy, precision, recall, and F1 on the validation set every epoch and perform a final evaluation on the test set after training. You can define the basic hyperparameters in the main function of the file, as well as other training parameters such as training, test, and validation set ratios, number of epochs, and batch sizes.
