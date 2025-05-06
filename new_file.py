import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sonar.inference_pipelines.text import (
    TextToEmbeddingModelPipeline,
    EmbeddingToTextModelPipeline,
)
import spacy
from sklearn.preprocessing import RobustScaler
from datasets import load_dataset
import subprocess

# Project Structure Setup
os.makedirs("large_concept_model/data/raw", exist_ok=True)
os.makedirs("large_concept_model/data/processed", exist_ok=True)
os.makedirs("large_concept_model/data/scripts", exist_ok=True)
os.makedirs("large_concept_model/models", exist_ok=True)
os.makedirs("large_concept_model/training", exist_ok=True)
os.makedirs("large_concept_model/inference", exist_ok=True)
os.makedirs("large_concept_model/evaluation", exist_ok=True)
os.makedirs("large_concept_model/notebooks", exist_ok=True)


# SONAR Utilities
class SonarEncoder:
    def __init__(self, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)
        self.model = TextToEmbeddingModelPipeline(
            encoder="text_sonar_basic_encoder",
            tokenizer="text_sonar_basic_encoder",
            device=device,
        )

    def encode(self, sentences, lang):
        return self.model.predict(sentences, source_lang=lang)


class SonarDecoder:
    def __init__(self, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)
        self.model = EmbeddingToTextModelPipeline(
            decoder="text_sonar_basic_decoder",
            tokenizer="text_sonar_basic_encoder",
            device=device,
        )

    def decode(self, embeddings, lang):
        return self.model.predict(embeddings, target_lang=lang)


# Base Model with common functionality
class BaseLCM(nn.Module):
    def __init__(
        self, d_model=1024, nhead=8, num_layers=12, dim_feedforward=2048, scaler=None
    ):
        super().__init__()
        self.d_model = d_model
        self.pos_encoder = nn.Embedding(50, d_model)  # Max seq len 50
        if scaler:
            self.register_buffer("median", torch.tensor(scaler.center_))
            self.register_buffer("scale", torch.tensor(scaler.scale_))
        else:
            self.median = None
            self.scale = None

    def normalize(self, x):
        if self.median is not None and self.scale is not None:
            return (x - self.median) / self.scale
        return x

    def denormalize(self, x):
        if self.median is not None and self.scale is not None:
            return x * self.scale + self.median
        return x

    def forward(self, src):
        # To be implemented by subclasses
        raise NotImplementedError("Subclasses must implement forward method")


# Standard autoregressive model with causal attention
class OneTowerLCM(BaseLCM):
    def __init__(
        self, d_model=1024, nhead=8, num_layers=12, dim_feedforward=2048, scaler=None
    ):
        super().__init__(d_model, nhead, num_layers, dim_feedforward, scaler)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output_layer = nn.Linear(d_model, d_model)

    def forward(self, src):
        # Ensure consistent dtype
        src = src.to(torch.float32)
        src = self.normalize(src)

        # Add positional encodings
        positions = (
            torch.arange(0, src.size(1), device=src.device)
            .unsqueeze(0)
            .repeat(src.size(0), 1)
        )
        src = src + self.pos_encoder(positions)

        # Create causal mask for autoregressive generation
        mask = nn.Transformer.generate_square_subsequent_mask(src.size(1)).to(
            device=src.device, dtype=torch.float32
        )

        # Forward pass through transformer
        output = self.transformer(src, mask=mask)
        output = self.output_layer(output)

        return self.denormalize(output)


# Two-tower architecture with separate encoder and predictor
class TwoTowerLCM(BaseLCM):
    def __init__(
        self, d_model=1024, nhead=8, num_layers=12, dim_feedforward=2048, scaler=None
    ):
        super().__init__(d_model, nhead, num_layers, dim_feedforward, scaler)

        # Encoder tower - bidirectional attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, batch_first=True
        )
        self.encoder_tower = nn.TransformerEncoder(encoder_layer, num_layers // 2)

        # Predictor tower - causal attention
        predictor_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, batch_first=True
        )
        self.predictor_tower = nn.TransformerEncoder(predictor_layer, num_layers // 2)

        self.output_layer = nn.Linear(d_model, d_model)
        self.intermediate_layer = nn.Linear(d_model, d_model)

        # Cross attention between towers
        self.cross_attention = nn.MultiheadAttention(d_model, nhead, batch_first=True)

    def forward(self, src):
        # Ensure consistent dtype
        src = src.to(torch.float32)
        src = self.normalize(src)

        # Add positional encodings
        positions = (
            torch.arange(0, src.size(1), device=src.device)
            .unsqueeze(0)
            .repeat(src.size(0), 1)
        )
        src_pos = src + self.pos_encoder(positions)

        # Encoder tower - bidirectional attention (no mask)
        encoder_output = self.encoder_tower(src_pos)
        encoder_output = self.intermediate_layer(encoder_output)

        # Predictor tower - causal attention
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            src_pos.size(1)
        ).to(device=src.device, dtype=torch.float32)
        predictor_output = self.predictor_tower(src_pos, mask=causal_mask)

        # Cross-attention between towers
        attn_output, _ = self.cross_attention(
            predictor_output, encoder_output, encoder_output
        )

        # Combine and project
        output = self.output_layer(attn_output + predictor_output)

        return self.denormalize(output)


# The original LCM model, maintained for backward compatibility
class LCMModel(OneTowerLCM):
    def __init__(
        self, d_model=1024, nhead=8, num_layers=12, dim_feedforward=2048, scaler=None
    ):
        super().__init__(d_model, nhead, num_layers, dim_feedforward, scaler)


# Function to initialize the appropriate model
def create_model(
    model_type, d_model=1024, nhead=8, num_layers=12, dim_feedforward=2048, scaler=None
):
    """
    Create and return the specified LCM model.

    Args:
        model_type (str): Type of model to create ('baselcm', 'onetower', or 'twotower')
        d_model (int): Dimension of the model
        nhead (int): Number of attention heads
        num_layers (int): Number of transformer layers
        dim_feedforward (int): Hidden dimension of feed-forward layers
        scaler: Scaler for embedding normalization

    Returns:
        The initialized model
    """
    model_type = model_type.lower()

    if model_type == "baselcm" or model_type == "base":
        return OneTowerLCM(d_model, nhead, num_layers, dim_feedforward, scaler)
    elif model_type == "onetower" or model_type == "one":
        return OneTowerLCM(d_model, nhead, num_layers, dim_feedforward, scaler)
    elif model_type == "twotower" or model_type == "two":
        return TwoTowerLCM(d_model, nhead, num_layers, dim_feedforward, scaler)
    else:
        raise ValueError(
            f"Unknown model type: {model_type}. Choose from 'baselcm', 'onetower', or 'twotower'."
        )


# Dataset
class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, seq_len=50):
        self.embeddings = embeddings
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.embeddings) - self.seq_len)

    def __getitem__(self, idx):
        return (
            self.embeddings[idx : idx + self.seq_len],
            self.embeddings[idx + 1 : idx + self.seq_len + 1],
        )


# Data Preparation with Hugging Face Datasets
def segment_sentences(text, lang="en"):
    """
    Segment text into sentences using SpaCy.

    Args:
        text (str): Text to segment
        lang (str): Language code for SpaCy (en, de, fr, etc.)
    """
    try:
        # Map language codes to SpaCy model names
        lang_mapping = {
            "eng": "en",
            "fra": "fr",
            "deu": "de",
            "spa": "es",
            "ita": "it",
            # Add more mappings as needed
        }

        # Get the first part of the language code (before underscore)
        if "_" in lang:
            lang_base = lang.split("_")[0]
        else:
            lang_base = lang

        # Map to SpaCy language code
        spacy_lang = lang_mapping.get(lang_base, lang_base)

        # Try to load the SpaCy model
        try:
            model_name = f"{spacy_lang}_core_web_sm"
            nlp = spacy.load(model_name)
        except OSError:
            # If model not found, download it first
            print(f"Downloading {model_name}...")
            subprocess.run(
                ["python", "-m", "spacy", "download", model_name], check=True
            )
            nlp = spacy.load(model_name)

        doc = nlp(text)
        return [sent.text for sent in doc.sents]
    except Exception as e:
        print(f"Error in sentence segmentation: {e}")
        # Fallback to simple splitting by periods
        return [s.strip() + "." for s in text.split(".") if s.strip()]


def prepare_data(
    dataset_name,
    config_name=None,
    split="train",
    text_column="text",
    lang="eng_Latn",
    output_file="large_concept_model/data/processed/embeddings.npy",
    device=None,
    batch_size=1000,
    sample_percentage=100,
):
    """
    Load a dataset from Hugging Face and prepare embeddings for LCM.

    Args:
        dataset_name (str): Name of the dataset on Hugging Face (e.g., 'wikitext').
        config_name (str): Name of the dataset configuration (e.g., 'wikitext-103-raw-v1').
        split (str): Dataset split to use (e.g., 'train', 'test').
        text_column (str): Name of the column containing text data.
        lang (str): Language code for SONAR (e.g., 'eng_Latn').
        output_file (str): Path to save the embeddings.
        device: Device to run encoding on (torch.device or str).
        batch_size (int): Number of texts to process per batch.
        sample_percentage (float): Percentage of the dataset to use (0-100).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    # Get base language for SpaCy
    spacy_lang = "en"  # Default to English
    if "_" in lang:
        lang_base = lang.split("_")[0]
        # Map common language codes
        if lang_base == "eng":
            spacy_lang = "en"
        elif lang_base == "fra":
            spacy_lang = "fr"
        # Add more mappings as needed

    print(f"Using device: {device}")
    print(f"Loading dataset: {dataset_name}, config: {config_name}, split: {split}")

    # Load dataset from Hugging Face
    try:
        dataset = load_dataset(dataset_name, config_name, split=split)

        # Sample a subset of the dataset if requested
        if sample_percentage < 100:
            sample_size = int(len(dataset) * (sample_percentage / 100))
            # Use random sampling without replacement
            sample_indices = np.random.choice(
                len(dataset), size=sample_size, replace=False
            )
            dataset = dataset.select(sample_indices)
            print(
                f"Sampled {sample_size} examples ({sample_percentage}% of original dataset)"
            )

        texts = dataset[text_column]
    except Exception as e:
        raise ValueError(
            f"Failed to load dataset '{dataset_name}' or access column '{text_column}': {str(e)}"
        )

    # Rest of the function remains the same
    encoder = SonarEncoder(device=device)
    all_embeddings = []

    # Process texts in batches for efficiency
    for i in range(0, len(texts), batch_size):
        print(f"Processing batch {i//batch_size + 1}/{(len(texts)//batch_size) + 1}")
        batch_texts = texts[i : i + batch_size]
        # Segment sentences using SpaCy
        sentences = [
            sent for text in batch_texts for sent in segment_sentences(text, spacy_lang)
        ]
        sentences.append("End of text.")  # Marker for end of sequence
        # Encode sentences into SONAR embeddings
        embeddings = encoder.encode(sentences, lang)
        all_embeddings.extend(embeddings.cpu().numpy())

    # Save embeddings
    np.save(output_file, np.array(all_embeddings))
    print(f"Embeddings saved to {output_file}")


# Training
def train(model, dataloader, epochs=10, lr=1e-4, device="cuda"):
    if isinstance(device, str):
        device = torch.device(device)

    model = model.to(device).type(torch.float32)  # Ensure float32
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"Training on {device} for {epochs} epochs")
    for epoch in range(epochs):
        total_loss = 0
        for src, tgt in dataloader:
            src = src.to(device=device, dtype=torch.float32)
            tgt = tgt.to(device=device, dtype=torch.float32)

            optimizer.zero_grad()
            pred = model(src)
            loss = criterion(pred, tgt)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(dataloader):.6f}")

    # Save the trained model
    torch.save(model.state_dict(), "large_concept_model/models/lcm_model.pt")
    print("Model saved to large_concept_model/models/lcm_model.pt")


# Inference
def generate(
    model, initial_sequence, eot_embedding, max_length=50, threshold=0.9, device=None
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    model = model.to(device)
    sequence = initial_sequence.clone().to(device)
    eot_embedding = eot_embedding.to(device)

    model.eval()
    with torch.no_grad():
        for _ in range(max_length):
            pred = model(sequence)
            next_emb = pred[:, -1, :]

            # Check if we're close to the end-of-text embedding
            sim_eot = torch.cosine_similarity(next_emb, eot_embedding, dim=-1)
            if sim_eot > threshold:
                print("End of text detected")
                break

            # Check if we're repeating ourselves
            if sequence.size(1) > 1:
                sim_prev = torch.cosine_similarity(next_emb, sequence[:, -1, :], dim=-1)
                if sim_prev > threshold:
                    print("Repetition detected")
                    break

            sequence = torch.cat([sequence, next_emb.unsqueeze(1)], dim=1)

    return sequence


# Main Execution
if __name__ == "__main__":
    # Direct variable assignment for Kaggle compatibility
    dataset_name = "wikitext"
    config_name = "wikitext-103-raw-v1"
    split = "train"
    text_column = "text"
    sample_percentage = 100
    epochs = 10
    batch_size = 8
    lr = 1e-4
    device = "cuda"
    model_type = "onetower"

    # Set device
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Download SpaCy model in advance
    try:
        spacy.load("en_core_web_sm")
    except OSError:
        print("Downloading SpaCy model...")
        subprocess.run(
            ["python", "-m", "spacy", "download", "en_core_web_sm"], check=True
        )

    # Prepare data
    prepare_data(
        dataset_name,
        config_name=config_name,
        split=split,
        text_column=text_column,
        device=device,
        sample_percentage=sample_percentage,
    )

    # Load embeddings
    embeddings_path = "large_concept_model/data/processed/embeddings.npy"
    embeddings = torch.tensor(np.load(embeddings_path), dtype=torch.float32)
    eot_embedding = embeddings[-1].unsqueeze(0)  # "End of text." embedding

    print(
        f"Loaded {embeddings.shape[0]} embeddings with dimension {embeddings.shape[1]}"
    )

    # Fit scaler
    print("Fitting robust scaler to normalize embeddings...")
    scaler = RobustScaler()
    scaler.fit(embeddings.cpu().numpy())

    # Initialize model
    print("Initializing model...")
    model = (
        create_model(model_type, d_model=embeddings.shape[1], scaler=scaler)
        .to(device)
        .type(torch.float32)
    )

    # Create dataset and dataloader
    print("Creating dataset and dataloader...")
    dataset = EmbeddingDataset(embeddings)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Train model
    print("Starting training...")
    train(model, dataloader, epochs=epochs, lr=lr, device=device)

    # Perform inference
    print("Performing inference...")
    decoder = SonarDecoder(device=device)
    initial_sequence = embeddings[:5].unsqueeze(0)  # Use first 5 embeddings as prompt
    print("Initial Sequence: ", initial_sequence)

    generated_sequence = generate(model, initial_sequence, eot_embedding, device=device)
    print("Generated Sequence: ", generated_sequence)

    generated_texts = decoder.decode(generated_sequence[0].cpu(), "eng_Latn")

    print("\nGenerated Text:")
    print(generated_texts)

    # Basic evaluation
    print("\nEvaluation: Check generated text coherence manually.")
