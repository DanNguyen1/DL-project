#!/usr/bin/env python
# coding: utf-8

from pathlib import Path
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score,
)
from model import Model
from dataset import create_df_from_dataset
from utils import batchify


def plot_res():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ## Load data from training
    training_path = Path(".")
    model_path = training_path / "model_checkpoint.pt"
    assert model_path.exists(), "Model checkpoint not found"
    results_path = training_path / "res.tar"
    assert results_path.exists(), "Training results file not found"
    figspath = training_path / "fig"
    figspath.mkdir(exist_ok=True)

    res_dict = torch.load(results_path, weights_only=False, map_location=device)

    epochs = np.arange(res_dict["epochs"])

    # ## Recreate original dataset
    dataset = Dataset.from_pandas(create_df_from_dataset())#.take(10)  # Take a subset of N datapoints for testing purposes
    dataset = dataset.train_test_split(
        test_size=res_dict["test_size"], seed=res_dict["seed"], shuffle=True
    )
    train_dataset = dataset["train"]
    train_dataset = train_dataset.train_test_split(
        test_size=(res_dict["val_size"] / (1 - res_dict["test_size"])),
        seed=res_dict["seed"],
        shuffle=True,
    )
    # val_dataset = train_dataset["test"]
    train_dataset = train_dataset["train"]
    test_dataset = dataset["test"]

    # ## Training results
    with plt.style.context("ggplot_perso.mplstyle"):
        fig, ax = plt.subplots()
        ax.plot(epochs, res_dict["losses"])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        fig.tight_layout()
        figpath = figspath / "loss.png"
        fig.savefig(fname=figpath)
        # plt.show()

    with plt.style.context("ggplot_perso.mplstyle"):
        fig, ax = plt.subplots()
        cmap = mpl.colormaps["tab20"].colors
        ax.plot(
            epochs, res_dict["accuracies_train"], label="Accuracy train", color=cmap[0]
        )
        ax.plot(epochs, res_dict["f1s_train"], label="F1 train", color=cmap[1])
        ax.plot(epochs, res_dict["accuracies_val"], label="Accuracy val", color=cmap[2])
        ax.plot(epochs, res_dict["f1s_val"], label="F1 val", color=cmap[3])
        ax.set_xlabel("Epoch")
        ax.legend(loc="lower right")
        # ax.legend(loc="best")
        fig.tight_layout()
        figpath = figspath / "accu_f1.png"
        fig.savefig(fname=figpath)
        # plt.show()

        # ## Model prediction on unseen data
    model_state_dict = torch.load(model_path, weights_only=False, map_location=device)
    model = Model(
        num_frames=res_dict["num_frames"],
        sample_rate=res_dict["sample_rate"],
        num_hidden_layers=res_dict["num_hidden_layers"],
        num_attention_heads=res_dict["num_attention_heads"],
        intermediate_size=res_dict["intermediate_size"],
    )
    model.load_state_dict(state_dict=model_state_dict)
    data = test_dataset
    model.eval()
    videos, audios = batchify(batch=data, sample_rate=res_dict["sample_rate"])
    y_score = torch.sigmoid(model(videos, audios)).squeeze(-1).detach().numpy()
    y_pred = np.round(y_score)
    # y_pred = torch.round(torch.sigmoid(model(videos, audios)).squeeze(-1)).detach().numpy()
    y_true = np.array(data["label"])

    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    f1 = f1_score(y_true=y_true, y_pred=y_pred)
    print(f"Accuracy: {accuracy}")
    print(f"F1: {f1}")

    # ## Confusion matrix
    ConfusionMatrixDisplay.from_predictions(
        y_true=y_true,
        y_pred=y_pred,
        cmap=plt.cm.Blues,
        # text_kw={"fontsize": 12}
    )
    plt.tight_layout()
    figpath = figspath / "confusion_matrix.png"
    plt.savefig(fname=figpath)
    # plt.show()

    # ## AUC ROC
    fpr, tpr, thresholds = roc_curve(y_true=y_true, y_score=y_score)
    roc_auc = roc_auc_score(y_true=y_true, y_score=y_score)
    with plt.style.context("ggplot_perso.mplstyle"):
        fig, ax = plt.subplots()
        ax.plot(
            fpr,
            tpr,
            label=f"ROC (AUC = {roc_auc:0.2f} )",
            lw=2,
            alpha=0.8,
        )
        ax.plot(
            np.array([0, 1]),
            np.array([0, 1]),
            label="Chance level",
            lw=2,
            color="grey",
            linestyle="dashed",
            alpha=0.8,
        )

        ax.set(
            xlabel="False Positive Rate",
            ylabel="True Positive Rate",
            # title="ROC curve",
        )
        # ax.legend(loc="lower right")
        ax.legend(loc="best")
        fig.tight_layout()
        figpath = figspath / "AUC_ROC.png"
        fig.savefig(fname=figpath)
        # plt.show()


if __name__ == "__main__":
    plot_res()
