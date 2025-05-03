import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_squared_error, cosine_similarity
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional, Union
import os
import json


def add_noise_to_embeddings(embeddings, noise_level=0.1):
    """Add Gaussian noise to embeddings.

    Args:
        embeddings: Input embeddings
        noise_level: Standard deviation of the noise

    Returns:
        torch.Tensor: Noisy embeddings
    """
    noise = torch.randn_like(embeddings) * noise_level
    return embeddings + noise


class GloveDataset(Dataset):
    """Dataset for handling embeddings with sequence structure.

    Attributes:
        embeddings: Input embeddings
        sequence_length: Length of sequences
        batch_size: Batch size for training
    """

    def __init__(self, embeddings, sequence_length, batch_size):
        """Initialize the dataset.

        Args:
            embeddings: Input embeddings
            sequence_length: Length of sequences
            batch_size: Batch size for training
        """
        self.embeddings = embeddings
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.num_samples = len(embeddings)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.embeddings[idx]


def compute_metrics(predictions, targets):
    """Compute evaluation metrics for model outputs.

    Args:
        predictions: Model predictions
        targets: Ground truth targets

    Returns:
        Dict: Dictionary of evaluation metrics
    """
    # Convert to numpy arrays for metric calculation
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # Calculate metrics
    mse = mean_squared_error(targets, predictions)

    # Calculate cosine similarity for each pair of vectors
    cos_sim_values = []
    for i in range(len(predictions)):
        cos_sim = cosine_similarity([predictions[i]], [targets[i]])[0][0]
        cos_sim_values.append(cos_sim)

    avg_cos_sim = np.mean(cos_sim_values)

    return {"mse": mse, "cosine_similarity": avg_cos_sim}


def plot_training_history(history, save_path=None):
    """Plot training history.

    Args:
        history: Dictionary containing training history
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Training Loss")
    if "val_loss" in history:
        plt.plot(history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()

    if "cosine_similarity" in history:
        plt.subplot(1, 2, 2)
        plt.plot(history["cosine_similarity"], label="Cosine Similarity")
        plt.xlabel("Epoch")
        plt.ylabel("Cosine Similarity")
        plt.title("Embedding Similarity")
        plt.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Training history plot saved to {save_path}")

    plt.show()


def setup_data_processing_pipeline(
    texts, encoder, lang="en", batch_size=32, max_texts=None
):
    """Setup data processing pipeline for text encoding.

    Args:
        texts: List of texts to encode
        encoder: Encoder model
        lang: Language code
        batch_size: Batch size for encoding
        max_texts: Maximum number of texts to process

    Returns:
        torch.Tensor: Encoded embeddings
    """
    if max_texts:
        texts = texts[:max_texts]

    print(f"Processing {len(texts)} texts...")
    embeddings = encoder.encode(texts, lang=lang, batch_size=batch_size)

    # Compute statistics for normalization
    mean = torch.mean(embeddings, dim=0)
    std = torch.std(embeddings, dim=0)

    print(
        f"Embeddings shape: {embeddings.shape}, Mean: {mean.mean().item():.4f}, Std: {std.mean().item():.4f}"
    )

    return embeddings, {"mean": mean, "std": std}


def split_train_test(data, test_size=0.1, shuffle=True):
    """Split data into training and test sets.

    Args:
        data: Data to split
        test_size: Fraction of data to use for testing
        shuffle: Whether to shuffle the data before splitting

    Returns:
        Tuple: (train_data, test_data)
    """
    if shuffle:
        indices = torch.randperm(len(data))
        data = data[indices]

    test_len = int(len(data) * test_size)
    train_data = data[:-test_len]
    test_data = data[-test_len:]

    return train_data, test_data


def save_config(config, path):
    """Save configuration to a JSON file.

    Args:
        config: Configuration dictionary
        path: Path to save the configuration
    """
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(path, "w") as f:
        json.dump(config, f, indent=4)

    print(f"Configuration saved to {path}")


def load_config(path):
    """Load configuration from a JSON file.

    Args:
        path: Path to load the configuration from

    Returns:
        Dict: Configuration dictionary
    """
    with open(path, "r") as f:
        config = json.load(f)

    return config
