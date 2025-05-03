import torch
from torch.utils.data import Dataset


def add_noise_to_embeddings(
    embeddings: torch.Tensor, noise_level: float = 0.1
) -> torch.Tensor:
    """
    Adds Gaussian noise to a tensor of embeddings.

    Args:
        embeddings (torch.Tensor): The input embeddings.
        noise_level (float): The standard deviation of the Gaussian noise.

    Returns:
        torch.Tensor: Embeddings with added noise.
    """
    noise = torch.randn_like(embeddings) * noise_level
    return embeddings + noise


class ConceptSequenceDataset(Dataset):
    """
    Dataset that provides sequences of concept embeddings and the next target embedding.
    """

    def __init__(self, embeddings: torch.Tensor, sequence_length: int):
        """
        Initializes the dataset.

        Args:
            embeddings (torch.Tensor): A tensor of shape (num_embeddings, embedding_dim).
            sequence_length (int): The length of the input sequence to provide to the model.
        """
        self.embeddings = embeddings.cpu()
        self.sequence_length = sequence_length

        if not isinstance(embeddings, torch.Tensor):
            raise TypeError(
                f"Embeddings must be a torch.Tensor, got {type(embeddings)}"
            )
        if embeddings.ndim != 2:
            raise ValueError(
                f"Embeddings tensor must be 2D (num_embeddings, dim), got shape {embeddings.shape}"
            )
        if len(embeddings) <= sequence_length:
            raise ValueError(
                f"Number of embeddings ({len(embeddings)}) must be greater than sequence_length ({sequence_length})"
            )

        self.embeddings = embeddings
        self.sequence_length = sequence_length
        # We can only create sequences up to the point where a target exists
        self.num_sequences = len(embeddings) - sequence_length

    def __len__(self) -> int:
        """Returns the number of possible sequences."""
        return self.num_sequences

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns an input sequence and the target (next) embedding.

        Args:
            idx (int): The starting index of the input sequence.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - input_sequence: Tensor of shape (sequence_length, embedding_dim)
                - target_embedding: Tensor of shape (embedding_dim)
        """
        if idx < 0 or idx >= self.num_sequences:
            raise IndexError(
                f"Index {idx} out of bounds for {self.num_sequences} sequences"
            )

        input_sequence = self.embeddings[idx : idx + self.sequence_length]
        target_embedding = self.embeddings[idx + self.sequence_length]
        return input_sequence, target_embedding
