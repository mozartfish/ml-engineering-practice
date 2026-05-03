# %% [markdown]
# ## Libraries

# %%
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# %% [markdown]
# ## Set up GPU

# %%
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Device: {device}")

# %% [markdown]
# ## Setup

# %%
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# %% [markdown]
# ## Data

# %% [markdown]
# ### KAGGLE Urban Sounds Dataset

# %%
import kagglehub

# Download latest version
download_path = Path(kagglehub.dataset_download("rupakroy/urban-sound-8k"))
root_dir = download_path / "UrbanSound8K" / "UrbanSound8K"
annotations_file = root_dir / "metadata" / "UrbanSound8K.csv"
audio_dir = root_dir / "audio"
print(f"download path: {download_path}\n")
# print(f"data contents: {list(download_path.iterdir())}\n")
# print(f"annotations: {annotations_file}\n")
# print(f"audio dir: {audio_dir}\n")
# print(f"audio dir contents: {list(audio_dir.iterdir())}\n")

# %%
df = pd.read_csv(annotations_file)
print(f"shape: {df.shape}")
print(f"column names: {df.columns}")
print(f"column types: {df.dtypes}")
print(f"classes: {df['class'].unique().tolist()}")

# %%
df.head()

# %% [markdown]
# ### UrbanSound Data Preprocessing

# %%
from torch.utils.data import Dataset
from torchcodec.decoders import AudioDecoder


class UrbanSoundDataset(Dataset):
    """
    annotations_file - all the annotations
    audio_dir - path to all the audio samples in the dataset
    """

    def __init__(self, annotations_file, audio_dir):
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = Path(audio_dir)

    """
    return number of samples in the dataset
    """

    def __len__(self):
        return len(self.annotations)

    """
    a_list[1] -> a_list.__getitem__(1)
    """

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)

        decoder = AudioDecoder(str(audio_sample_path))
        signal = decoder.get_all_samples()
        sr = signal.sample_rate
        samples = signal.data

        return samples, label

    """
    """

    def _get_audio_sample_path(self, index):
        row = self.annotations.iloc[index]
        fold = f"fold{row['fold']}"  # named col instead of [index, 5]
        return (
            self.audio_dir / fold / row["slice_file_name"]
        )  # pathlib / instead of os.path.join

    """
    """

    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index][
            "classID"
        ]  # named col instead of [index, 6]


# %%
usd = UrbanSoundDataset(annotations_file, audio_dir)
print(f"There are {len(usd)} samples in the dataset.")
signal, label = usd[0]
print(f"Signal shape: {signal.shape} | Label: {label}")

# %% [markdown]
# ## Dataset Processing

# %% [markdown]
# ### UrbanSound Dataset Preprocessing

# %% [markdown]
# #### Data Loading

# %% [markdown]
# #### Create Dataset + DataLoader

# %% [markdown]
# ## Neural Network Architecture - Feed-Forward Neural Network (FFNN)


# %%
# neural network architecture
class MNISTFeedForward(nn.Module):
    def __init__(self, num_classes=10, dropout_prob=0.3):
        super().__init__()
        # build a model
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def train(
    model, train_dataloader, valid_dataloader, optimizer, loss_fn, device, epochs=50
):
    history = defaultdict(list)
    for epoch in range(epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for x_batch, y_batch in train_dataloader:
            # get a sample of data
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            # set the gradients to 0
            optimizer.zero_grad()

            # run forward pass
            preds = model(x_batch)

            # run cross entropy loss
            loss = loss_fn(preds, y_batch)

            # run backward pass
            loss.backward()

            # update parameters
            optimizer.step()

            # accumulate training loss
            train_loss += loss.item() * x_batch.size(0)

            # number of correct predictions in batch
            train_correct += (preds.argmax(1) == y_batch).sum().item()

            # number of samples in the batch
            train_total += y_batch.size(0)

        # check the model accuracy once per epoch on the validation data
        valid_loss, valid_acc = evaluate(model, valid_dataloader, loss_fn, device)

        # update the training statistics
        history["train_loss"].append(train_loss / train_total)
        history["train_acc"].append(train_correct / train_total)
        history["valid_loss"].append(valid_loss)
        history["valid_acc"].append(valid_acc)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"train_loss: {train_loss / train_total:.4f} - train_acc: {train_correct / train_total:.4f} | "
            f"valid_loss: {valid_loss:.4f} - valid_acc: {valid_acc:.4f}"
        )

    print("---------------------------------------")
    print("Training Finished!\n")

    return history


def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            preds = model(x_batch)
            loss = loss_fn(preds, y_batch)
            total_loss += loss.item() * x_batch.size(0)
            correct += (preds.argmax(1) == y_batch).sum().item()
            total += y_batch.size(0)

    return total_loss / total, correct / total


# %% [markdown]
# ## Overfitting Debugging


# %%
def plot_history(history):
    fig, axs = plt.subplots(2, figsize=(10, 8))

    # create accuracy subplot
    axs[0].plot(history["train_acc"], label="train accuracy")
    axs[0].plot(history["valid_acc"], label="valid accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="best")
    axs[0].set_title("Accuracy eval")

    # create error subplot
    axs[1].plot(history["train_loss"], label="train error")
    axs[1].plot(history["valid_loss"], label="valid error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="best")
    axs[1].set_title("Error eval")

    fig.tight_layout()
    plt.show()


# %% [markdown]
# ## Neural Networks go brrrrrr......

# %% [markdown]
# ### Training Setup

# %%
# # # compile network
# # model = MusicGenreClassifier().to(device)

# # # weight_decay = l2 regularization in pytorch
# # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)

# # # loss function
# # loss_fn = nn.CrossEntropyLoss()

# # # train network
# # model_training_history = train(model, train_dataloader, valid_dataloader, optimizer, loss_fn, device, epochs=50)

# model = MNISTFeedForward().to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# loss_fn = nn.CrossEntropyLoss()

# model_training_history = train(
#     model, train_dataloader, valid_dataloader,
#     optimizer, loss_fn, device, epochs=10
# )

# %% [markdown]
# ### Overfitting Debugging

# %%
# # plot accuracy and error over the epochs
# plot_history(model_training_history)

# %% [markdown]
# ### Evaluation

# %%
# test_loss, test_acc = evaluate(model, test_dataloader, loss_fn, device)
# print(f"Test loss: {test_loss:.4f} | Test accuracy: {test_acc:.4f}")

# test_loss, test_acc = evaluate(model, test_dataloader, loss_fn, device)
# print(f"Test loss: {test_loss:.4f} | Test accuracy: {test_acc:.4f}")

# %% [markdown]
# ## Inference

# %% [markdown]
# ### GTZAN Inference

# %%
# def predict(model, X, y):
#     X = X.unsqueeze(0).to(device)
#     model.eval()
#     with torch.no_grad():
#         logits = model(X)
#         predicted_class = logits.argmax(1).item()

#     target_genre = mapping[y.item()]
#     predicted_genre = mapping[predicted_class]
#     print(f"Prediction: {predicted_class} - {predicted_genre} | Target: {y.item()} - {target_genre}")

# %%
# X, y = test_dataset[100]
# predict(model, X, y)

# %% [markdown]
# ### MNIST Inference

# %%
# def predict(model, X, y):
#     class_mapping = [str(i) for i in range(10)]
#     X = X..unsqueeze(0).to(device)
#     model.eval()
#     with torch.no_grad():
#         logits = model(X)   # add batch dim
#         predicted = class_mapping[logits.argmax(1).item()]
#         expected  = class_mapping[y]
#     print(f"Predicted: '{predicted}' | Expected: '{expected}'")

# %%
# X, y = test_data[0]
# predict(model, X, y)

# %% [markdown]
# ### UrbanSound Inference

# %%
