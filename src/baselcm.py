import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from transformers.models.m2m_100.modeling_m2m_100 import M2M100Encoder
from tqdm.auto import tqdm
import os


class SonarEncoder:
    """Text encoder using SONAR model for generating embeddings.

    Attributes:
        encoder: The M2M100Encoder model for encoding text
        tokenizer: Tokenizer for the encoder
        device: Device to run the model on (cuda or cpu)
    """

    def __init__(self, model_name="cointegrated/SONAR_200_text_encoder", device="cpu"):
        """Initialize the SonarEncoder.

        Args:
            model_name: Name of the model to use for encoding
            device: Device to run the model on
        """
        self.encoder = M2M100Encoder.from_pretrained(model_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = device

    def encode(self, texts, lang, batch_size=32, norm=False):
        """Encode texts into embedding vectors.

        Args:
            texts: List of texts to encode
            lang: Language code for the texts
            batch_size: Batch size for encoding
            norm: Whether to normalize the embeddings

        Returns:
            torch.Tensor: Embeddings for the input texts
        """
        if self.tokenizer is None or self.encoder is None:
            raise ValueError("Tokenizer or encoder is not initialized.")

        self.tokenizer.src_lang = lang
        texts = texts if isinstance(texts, list) else [texts]

        embeddings = []
        with torch.inference_mode():
            for i in tqdm(
                range(0, len(texts), batch_size), desc="Encoding Batches", unit="batch"
            ):
                batch_texts = texts[i : i + batch_size]
                batch = self.tokenizer(
                    batch_texts, return_tensors="pt", padding=True, truncation=True
                ).to(self.device)
                seq_embs = self.encoder(**batch).last_hidden_state
                mask = batch.attention_mask

                # Compute mean embedding for each sequence
                mean_emb = (seq_embs * mask.unsqueeze(-1)).sum(1) / mask.unsqueeze(
                    -1
                ).sum(1)
                if norm:
                    mean_emb = F.normalize(mean_emb, dim=1)

                embeddings.append(mean_emb)

        return torch.cat(embeddings, dim=0)


class PreNet(nn.Module):
    """Preprocessing network for input embeddings.

    Attributes:
        linear: Linear layer for transforming input
        scaler_mean: Mean value for normalization
        scaler_std: Standard deviation for normalization
    """

    def __init__(self, input_dim, hidden_dim):
        """Initialize the PreNet.

        Args:
            input_dim: Dimension of input embeddings
            hidden_dim: Dimension of hidden layer
        """
        super(PreNet, self).__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(0.1)
        self.activation = nn.GELU()
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.scaler_mean = 0.0
        self.scaler_std = 1.0

    def normalize(self, x):
        """Normalize the input.

        Args:
            x: Input tensor

        Returns:
            torch.Tensor: Normalized input
        """
        return (x - self.scaler_mean) / self.scaler_std

    def forward(self, x):
        """Forward pass.

        Args:
            x: Input tensor

        Returns:
            torch.Tensor: Processed input
        """
        x = self.normalize(x)
        x = self.linear(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.layer_norm(x)
        return x


class PostNet(nn.Module):
    """Post-processing network for output embeddings.

    Attributes:
        linear: Linear layer for transforming output
        scaler_mean: Mean value for denormalization
        scaler_std: Standard deviation for denormalization
    """

    def __init__(self, hidden_dim, output_dim):
        """Initialize the PostNet.

        Args:
            hidden_dim: Dimension of hidden layer
            output_dim: Dimension of output embeddings
        """
        super(PostNet, self).__init__()
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.activation = nn.GELU()
        self.scaler_mean = 0.0
        self.scaler_std = 1.0

    def denormalize(self, x):
        """Denormalize the output.

        Args:
            x: Output tensor

        Returns:
            torch.Tensor: Denormalized output
        """
        return x * self.scaler_std + self.scaler_mean

    def forward(self, x):
        """Forward pass.

        Args:
            x: Input tensor

        Returns:
            torch.Tensor: Processed output
        """
        x = self.layer_norm(x)
        x = self.activation(x)
        x = self.linear(x)
        x = self.denormalize(x)
        return x


class TransformerDecoder(nn.Module):
    """Transformer decoder for sequence modeling.

    Attributes:
        layers: List of transformer decoder layers
        pos_encoder: Positional encoding for input sequences
        causal_mask: Mask to ensure causal attention
    """

    def __init__(
        self, hidden_dim, num_heads, num_layers, ff_dim, dropout=0.1, max_seq_len=512
    ):
        """Initialize the TransformerDecoder.

        Args:
            hidden_dim: Dimension of hidden layer
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            ff_dim: Dimension of feedforward network
            dropout: Dropout probability
            max_seq_len: Maximum sequence length
        """
        super(TransformerDecoder, self).__init__()
        # Add mask to ensure causal attention
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool(),
        )

        self.layers = nn.ModuleList(
            [
                nn.TransformerDecoderLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dim_feedforward=ff_dim,
                    dropout=dropout,
                    batch_first=True,
                )
                for _ in range(num_layers)
            ]
        )

        self.pos_encoder = nn.Parameter(torch.zeros(1, max_seq_len, hidden_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Forward pass.

        Args:
            x: Input tensor

        Returns:
            torch.Tensor: Processed sequence
        """
        seq_len = x.size(1)
        # Add positional encodings
        pos_enc = self.pos_encoder[:, :seq_len, :]
        x = x + pos_enc
        x = self.dropout(x)

        mask = self.causal_mask[:seq_len, :seq_len]
        for layer in self.layers:
            x = layer(x, x, tgt_mask=mask)
        return x


class BaseLCM(nn.Module):
    """Base Latent Consistency Model.

    This model processes input embeddings through a transformer-based architecture
    to generate output embeddings that match the target distribution.

    Attributes:
        prenet: Preprocessing network
        transformer_decoder: Main transformer component
        postnet: Post-processing network
    """

    def __init__(
        self, input_dim, hidden_dim, num_heads, num_layers, ff_dim, output_dim
    ):
        """Initialize the BaseLCM.

        Args:
            input_dim: Dimension of input embeddings
            hidden_dim: Dimension of hidden layer
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            ff_dim: Dimension of feedforward network
            output_dim: Dimension of output embeddings
        """
        super(BaseLCM, self).__init__()
        self.prenet = PreNet(input_dim, hidden_dim)
        self.transformer_decoder = TransformerDecoder(
            hidden_dim, num_heads, num_layers, ff_dim
        )
        self.postnet = PostNet(hidden_dim, output_dim)

    def forward(self, x):
        """Forward pass through the model.

        Args:
            x: Input tensor, shape (batch_size, [seq_len], input_dim)

        Returns:
            torch.Tensor: Output embeddings
        """
        # Add sequence dimension if not present
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        x = self.prenet(x)
        x = self.transformer_decoder(x)
        x = self.postnet(x)
        return x.squeeze(1)  # Remove sequence dimension if single step

    def save(self, path):
        """Save the model to a file.

        Args:
            path: Path to save the model
        """
        directory = os.path.dirname(path)
        os.makedirs(directory, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "prenet_mean": self.prenet.scaler_mean,
                "prenet_std": self.prenet.scaler_std,
                "postnet_mean": self.postnet.scaler_mean,
                "postnet_std": self.postnet.scaler_std,
            },
            path,
        )
        print(f"Model saved to {path}")

    @classmethod
    def load(
        cls,
        path,
        input_dim,
        hidden_dim,
        num_heads,
        num_layers,
        ff_dim,
        output_dim,
        device="cpu",
    ):
        """Load the model from a file.

        Args:
            path: Path to load the model from
            input_dim: Dimension of input embeddings
            hidden_dim: Dimension of hidden layer
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            ff_dim: Dimension of feedforward network
            output_dim: Dimension of output embeddings
            device: Device to load the model on

        Returns:
            BaseLCM: Loaded model
        """
        model = cls(input_dim, hidden_dim, num_heads, num_layers, ff_dim, output_dim)
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        # Load scaling parameters
        model.prenet.scaler_mean = checkpoint.get("prenet_mean", 0.0)
        model.prenet.scaler_std = checkpoint.get("prenet_std", 1.0)
        model.postnet.scaler_mean = checkpoint.get("postnet_mean", 0.0)
        model.postnet.scaler_std = checkpoint.get("postnet_std", 1.0)

        model.to(device)
        model.eval()
        return model
