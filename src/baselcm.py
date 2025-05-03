import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from transformers.models.m2m_100.modeling_m2m_100 import M2M100Encoder, M2M100Decoder
from tqdm.auto import tqdm
from typing import List, Optional


class SonarEncoder:
    """
    Encodes text into sentence embeddings using a pre-trained SONAR model (M2M100Encoder).
    """

    def __init__(
        self,
        model_name: str = "cointegrated/SONAR_200_text_encoder",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initializes the SonarEncoder.

        Args:
            model_name (str): The name of the pre-trained SONAR encoder model on Hugging Face Hub.
            device (str): The device ('cpu' or 'cuda') to run the model on.
        """
        self.encoder = M2M100Encoder.from_pretrained(model_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = device
        print(f"SonarEncoder initialized on device: {self.device}")

    def encode(
        self, texts: List[str], lang: str, batch_size: int = 32, norm: bool = False
    ) -> torch.Tensor:
        """
        Encodes a list of texts into embeddings.

        Args:
            texts (List[str]): A list of sentences or texts to encode.
            lang (str): The language code for the tokenizer (e.g., 'en').
            batch_size (int): The number of texts to process in each batch.
            norm (bool): Whether to normalize the resulting embeddings.

        Returns:
            torch.Tensor: A tensor containing the sentence embeddings.
        """
        if self.tokenizer is None or self.encoder is None:
            raise ValueError("Tokenizer or encoder is not initialized.")

        self.tokenizer.src_lang = lang

        all_embeddings = []
        self.encoder.eval()  # Ensure model is in evaluation mode
        with torch.inference_mode():
            for i in tqdm(
                range(0, len(texts), batch_size), desc="Encoding Batches", unit="batch"
            ):
                batch_texts = texts[i : i + batch_size]
                try:
                    batch = self.tokenizer(
                        batch_texts, return_tensors="pt", padding=True, truncation=True
                    ).to(self.device)
                    seq_embs = self.encoder(**batch).last_hidden_state
                    mask = batch.attention_mask

                    # Compute mean embedding for each sequence
                    # Mask out padding tokens before averaging
                    masked_embs = seq_embs * mask.unsqueeze(-1)
                    sum_embs = masked_embs.sum(1)
                    seq_len = mask.sum(1).unsqueeze(-1)
                    mean_emb = sum_embs / seq_len.clamp(
                        min=1e-9
                    )  # Avoid division by zero for empty sequences

                    if norm:
                        mean_emb = F.normalize(mean_emb, dim=1, p=2)

                    all_embeddings.append(
                        mean_emb.to(self.device)
                    )  # Keep on GPU if device is GPU
                except Exception as e:
                    print(f"Error encoding batch {i // batch_size}: {e}")
                    print(
                        f"Problematic texts: {batch_texts[:2]}..."
                    )  # Print first few problematic texts
                    continue  # Skip this batch

        if not all_embeddings:
            return torch.empty(
                (0, self.encoder.config.d_model),
                dtype=torch.float32,
                device=self.device,
            )

        return torch.cat(all_embeddings, dim=0)


class SonarDecoder:
    """
    Decodes sentence embeddings back to text using a pre-trained SONAR decoder model.
    """

    def __init__(
        self,
        model_name: str = "facebook/m2m100_418M",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initializes the SonarDecoder.

        Args:
            model_name (str): The name of the pre-trained M2M100 model on Hugging Face Hub.
            device (str): The device ('cpu' or 'cuda') to run the model on.
        """
        from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

        self.model = M2M100ForConditionalGeneration.from_pretrained(model_name).to(
            device
        )
        self.tokenizer = M2M100Tokenizer.from_pretrained(model_name)
        self.device = device
        print(f"SonarDecoder initialized on device: {self.device}")

    def decode(
        self,
        embeddings: torch.Tensor,
        tgt_lang: str,
        max_length: int = 100,
        num_beams: int = 4,
    ) -> List[str]:
        """
        Decodes embeddings back to text.

        Args:
            embeddings (torch.Tensor): Tensor of embeddings to decode.
            tgt_lang (str): Target language code (e.g., 'en').
            max_length (int): Maximum length of generated text.
            num_beams (int): Number of beams for beam search.

        Returns:
            List[str]: Decoded sentences.
        """
        self.tokenizer.tgt_lang = tgt_lang

        # Move embeddings to decoder's device if needed
        embeddings = embeddings.to(self.device)

        # Set decoder_start_token_id based on target language
        self.model.config.decoder_start_token_id = self.tokenizer.get_lang_id(tgt_lang)

        # Generate text from embeddings using beam search
        decoded_text = []
        with torch.inference_mode():
            for emb in tqdm(embeddings, desc="Decoding", unit="embedding"):
                # Reshape embedding for model
                encoder_hidden_states = emb.unsqueeze(0).unsqueeze(0)

                # Generate text
                output_ids = self.model.generate(
                    encoder_outputs=[encoder_hidden_states],
                    max_length=max_length,
                    num_beams=num_beams,
                    early_stopping=True,
                )

                # Decode the generated token IDs into text
                text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
                decoded_text.append(text)

        return decoded_text


class PreNet(nn.Module):
    """
    A pre-network that normalizes input embeddings and maps them to the model's hidden dimension.
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super(PreNet, self).__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        # Note: These scalers are currently fixed. For data-dependent normalization,
        # they should be calculated from the training data statistics.
        self.scaler_mean = 0.0
        self.scaler_std = 1.0

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Applies normalization."""
        return (x - self.scaler_mean) / self.scaler_std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the PreNet."""
        x = self.normalize(x)
        x = self.linear(x)
        return x


class PostNet(nn.Module):
    """
    A post-network that maps hidden representations back to the original embedding space
    and applies de-normalization.
    """

    def __init__(self, hidden_dim: int, output_dim: int):
        super(PostNet, self).__init__()
        self.linear = nn.Linear(hidden_dim, output_dim)
        # Note: These scalers are currently fixed and should match PreNet's inverse.
        self.scaler_mean = 0.0
        self.scaler_std = 1.0

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Applies de-normalization."""
        return x * self.scaler_std + self.scaler_mean

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the PostNet."""
        x = self.linear(x)
        x = self.denormalize(x)
        return x


class TransformerDecoder(nn.Module):
    """
    A standard Transformer Decoder stack with causal masking for sequence generation.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        ff_dim: int,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ):
        super(TransformerDecoder, self).__init__()
        self.max_seq_len = max_seq_len
        # Register buffer for causal mask to ensure it's moved to the correct device
        self.register_buffer(
            "causal_mask",
            torch.triu(
                torch.ones(max_seq_len, max_seq_len, dtype=torch.bool), diagonal=1
            ),
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,  # Ensure input/output format is (batch, seq, feature)
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

        # Positional encoding parameter
        self.pos_encoder = nn.Parameter(torch.zeros(1, max_seq_len, hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Transformer Decoder.

        Args:
            x (torch.Tensor): Input sequence tensor of shape (batch_size, seq_len, hidden_dim).

        Returns:
            torch.Tensor: Output sequence tensor of shape (batch_size, seq_len, hidden_dim).
        """
        batch_size, seq_len, _ = x.shape
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds maximum positional encoding length {self.max_seq_len}"
            )

        # Add positional encodings
        pos_enc = self.pos_encoder[:, :seq_len, :]
        x = x + pos_enc

        # Ensure the mask is the correct size and on the correct device
        mask = self.causal_mask[:seq_len, :seq_len].to(x.device)

        # Pass through decoder layers
        # Note: In standard TransformerDecoderLayer, tgt is the input sequence,
        # memory is usually the encoder output (not used here, self-attention only).
        # tgt_mask prevents attending to future positions.
        for layer in self.layers:
            # The layer expects memory, but for decoder-only models, we pass tgt as memory.
            # tgt_mask is used for causal attention within the sequence 'x'.
            x = layer(tgt=x, memory=x, tgt_mask=mask)
        return x


class BaseLCM(nn.Module):
    """
    Base Latent Concept Model (BaseLCM) implementation.
    Connects PreNet, TransformerDecoder, and PostNet.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        ff_dim: int,
        output_dim: int,
        max_seq_len: int = 512,
    ):
        super(BaseLCM, self).__init__()
        self.prenet = PreNet(input_dim, hidden_dim)
        self.transformer_decoder = TransformerDecoder(
            hidden_dim, num_heads, num_layers, ff_dim, max_seq_len=max_seq_len
        )
        self.postnet = PostNet(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the BaseLCM model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim).
                              If shape is (batch_size, input_dim), it's unsqueezed to (batch_size, 1, input_dim).

        Returns:
            torch.Tensor: Output tensor representing predicted embeddings.
                          Shape is (batch_size, seq_len, output_dim).
                          If input was unsqueezed, output is squeezed back to (batch_size, output_dim).
        """
        # Add sequence dimension if input is single step (batch_size, input_dim)
        squeezed_output = False
        if x.ndim == 2:
            x = x.unsqueeze(1)
            squeezed_output = True

        # Pass through the components
        x = self.prenet(x)
        x = self.transformer_decoder(x)
        x = self.postnet(x)

        # Remove sequence dimension if input was single step
        if squeezed_output:
            x = x.squeeze(1)
        return x
