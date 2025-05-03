import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from transformers.models.m2m_100.modeling_m2m_100 import M2M100Encoder
from tqdm.auto import tqdm
import os


class SonarEncoder:
    """Text encoder using SONAR model for generating concept embeddings.

    Following Meta's LCM architecture, this encoder transforms text into concept embeddings.
    SONAR supports 200 languages as text input and output.
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
        """Encode texts into concept embedding vectors.

        Args:
            texts: List of texts to encode
            lang: Language code for the texts
            batch_size: Batch size for encoding
            norm: Whether to normalize the embeddings

        Returns:
            torch.Tensor: Concept embeddings for the input texts
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

                # Compute mean embedding for each sequence to get concept representation
                mean_emb = (seq_embs * mask.unsqueeze(-1)).sum(1) / mask.unsqueeze(
                    -1
                ).sum(1)
                if norm:
                    mean_emb = F.normalize(mean_emb, dim=1)

                embeddings.append(mean_emb)

        return torch.cat(embeddings, dim=0)


class PreNet(nn.Module):
    """Preprocessing network for input concept embeddings.

    As described in Meta's LCM paper, PreNet normalizes the concept embeddings
    from SONAR and maps them into the Transformer's dimension.
    """

    def __init__(self, input_dim, hidden_dim, dropout_rate=0.1):
        """Initialize the PreNet.

        Args:
            input_dim: Dimension of input embeddings from SONAR
            hidden_dim: Dimension of Transformer model
            dropout_rate: Dropout rate for regularization
        """
        super(PreNet, self).__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.GELU()
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.scaler_mean = 0.0
        self.scaler_std = 1.0

    def normalize(self, x):
        """Normalize the input concept embeddings.

        Args:
            x: Input tensor

        Returns:
            torch.Tensor: Normalized input
        """
        return (x - self.scaler_mean) / self.scaler_std

    def forward(self, x):
        """Forward pass through the PreNet.

        Args:
            x: Input concept embeddings

        Returns:
            torch.Tensor: Processed input for the Transformer
        """
        x = self.normalize(x)
        x = self.linear(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.layer_norm(x)
        return x


class PostNet(nn.Module):
    """Post-processing network for output concept embeddings.

    As described in Meta's LCM paper, PostNet projects the model output
    back to SONAR's dimension.
    """

    def __init__(self, hidden_dim, output_dim, dropout_rate=0.1):
        """Initialize the PostNet.

        Args:
            hidden_dim: Dimension of Transformer output
            output_dim: Dimension of SONAR embedding space
            dropout_rate: Dropout rate for regularization
        """
        super(PostNet, self).__init__()
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.GELU()
        self.scaler_mean = 0.0
        self.scaler_std = 1.0

    def denormalize(self, x):
        """Denormalize the output.

        Args:
            x: Output tensor

        Returns:
            torch.Tensor: Denormalized output in SONAR embedding space
        """
        return x * self.scaler_std + self.scaler_mean

    def forward(self, x):
        """Forward pass through the PostNet.

        Args:
            x: Transformer output tensor

        Returns:
            torch.Tensor: Processed output in SONAR embedding space
        """
        x = self.layer_norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear(x)
        x = self.denormalize(x)
        return x


class TransformerDecoder(nn.Module):
    """Transformer decoder for concept sequence processing.

    According to Meta's LCM paper, this is the main component that performs
    reasoning on the sequence of concepts.
    """

    def __init__(
        self, hidden_dim, num_heads, num_layers, ff_dim, dropout=0.1, max_seq_len=512
    ):
        """Initialize the TransformerDecoder.

        Args:
            hidden_dim: Dimension of hidden layers
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            ff_dim: Dimension of feedforward network
            dropout: Dropout probability
            max_seq_len: Maximum sequence length for positional encoding
        """
        super(TransformerDecoder, self).__init__()
        # Add mask to ensure causal attention (only attend to previous concepts)
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool(),
        )

        # Create stacked transformer decoder layers
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

        # Learnable positional encoding
        self.pos_encoder = nn.Parameter(torch.zeros(1, max_seq_len, hidden_dim))
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        """Forward pass through the Transformer decoder.

        Args:
            x: Input concept embedding sequence

        Returns:
            torch.Tensor: Processed concept sequence
        """
        seq_len = x.size(1)
        # Add positional encodings
        pos_enc = self.pos_encoder[:, :seq_len, :]
        x = x + pos_enc
        x = self.dropout(x)

        # Apply causal attention mask to ensure auto-regressive behavior
        mask = self.causal_mask[:seq_len, :seq_len]

        # Process through transformer layers
        for layer in self.layers:
            x = layer(x, x, tgt_mask=mask)

        x = self.layer_norm(x)
        return x


class BaseLCM(nn.Module):
    """Base Latent Consistency Model (LCM) as described in Meta's paper.

    This model processes concept embeddings through a transformer-based architecture
    to generate the next concept in the sequence, similar to how LLMs predict
    the next token but operating in the embedding space.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_heads,
        num_layers,
        ff_dim,
        output_dim,
        dropout_rate=0.1,
    ):
        """Initialize the BaseLCM.

        Args:
            input_dim: Dimension of SONAR concept embeddings
            hidden_dim: Dimension of hidden layers in transformer
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            ff_dim: Dimension of feedforward network
            output_dim: Dimension of output concept embeddings
            dropout_rate: Dropout rate for regularization
        """
        super(BaseLCM, self).__init__()
        self.prenet = PreNet(input_dim, hidden_dim, dropout_rate)
        self.transformer_decoder = TransformerDecoder(
            hidden_dim, num_heads, num_layers, ff_dim, dropout=dropout_rate
        )
        self.postnet = PostNet(hidden_dim, output_dim, dropout_rate)

    def forward(self, x):
        """Forward pass through the BaseLCM.

        Args:
            x: Input concept embeddings, shape (batch_size, [seq_len], input_dim)

        Returns:
            torch.Tensor: Generated next concept embedding
        """
        # Add sequence dimension if not present (for single concept input)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        # Process through model components
        x = self.prenet(x)
        x = self.transformer_decoder(x)
        x = self.postnet(x)

        # Return the last concept in the sequence (the prediction)
        return x[:, -1]

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


class DiffusionLCM(BaseLCM):
    """Diffusion-based LCM as described in Meta's paper.

    This extends BaseLCM with diffusion capabilities for more flexible
    concept generation, similar to diffusion models used in image generation.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_heads,
        num_layers,
        ff_dim,
        output_dim,
        diffusion_steps=10,
        dropout_rate=0.1,
    ):
        """Initialize the DiffusionLCM.

        Args:
            input_dim: Dimension of SONAR concept embeddings
            hidden_dim: Dimension of hidden layers in transformer
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            ff_dim: Dimension of feedforward network
            output_dim: Dimension of output concept embeddings
            diffusion_steps: Number of diffusion steps
            dropout_rate: Dropout rate for regularization
        """
        super(DiffusionLCM, self).__init__(
            input_dim,
            hidden_dim,
            num_heads,
            num_layers,
            ff_dim,
            output_dim,
            dropout_rate,
        )
        self.diffusion_steps = diffusion_steps

        # Timestep embedding projector for diffusion
        self.time_embedder = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )

    def add_noise(self, x, noise_level):
        """Add noise to concept embeddings.

        Args:
            x: Clean concept embeddings
            noise_level: Noise intensity (0 to 1)

        Returns:
            torch.Tensor: Noisy concept embeddings
        """
        noise = torch.randn_like(x) * noise_level
        return x + noise

    def denoise_step(self, x, context, timestep):
        """Perform one denoising step.

        Args:
            x: Noisy concept embedding
            context: Context concept embeddings
            timestep: Current diffusion timestep (0 to diffusion_steps)

        Returns:
            torch.Tensor: Denoised concept embedding
        """
        # Create full input with context and noisy target
        full_input = torch.cat([context, x.unsqueeze(1)], dim=1)

        # Embed timestep
        t_emb = self.time_embedder(
            (timestep / self.diffusion_steps).view(-1, 1).float()
        )

        # Process through model
        x_pre = self.prenet(full_input)

        # Add timestep embedding
        x_pre[:, -1] = x_pre[:, -1] + t_emb

        x_trans = self.transformer_decoder(x_pre)
        output = self.postnet(x_trans)

        # Return only the prediction (last position)
        return output[:, -1]

    def generate(self, context, steps=None):
        """Generate next concept through iterative denoising.

        Args:
            context: Context concept embeddings
            steps: Number of diffusion steps (defaults to self.diffusion_steps)

        Returns:
            torch.Tensor: Generated clean concept embedding
        """
        steps = steps or self.diffusion_steps
        batch_size = context.shape[0]
        emb_dim = context.shape[-1]
        device = context.device

        # Start from pure noise
        x_t = torch.randn(batch_size, emb_dim).to(device)

        # Iteratively denoise
        for t in range(steps - 1, -1, -1):
            timestep = torch.ones(batch_size, device=device) * t
            x_t = self.denoise_step(x_t, context, timestep)

            # Add a small amount of noise except in the last step
            if t > 0:
                noise_scale = 0.1 * (t / steps)
                x_t = x_t + torch.randn_like(x_t) * noise_scale

        return x_t
