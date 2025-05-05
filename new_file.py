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

# Project Structure Setup
os.makedirs("large_concept_model/data/raw", exist_ok=True)
os.makedirs("large_concept_model/data/processed", exist_ok=True)
os.makedirs("large_concept_model/data/scripts", exist_ok=True)
os.makedirs("large_concept_model/models", exist_ok=True)
os.makedirs("large_concept_model/training", exist_ok=True)
os.makedirs("large_concept_model/inference", exist_ok=True)
os.makedirs("large_concept_model/evaluation", exist_ok=True)
os.makedirs("large_concept_model/notebooks", exist_ok=True)

# Set device to GPU by default if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


# LCM Model
class LCMModel(nn.Module):
    def __init__(
        self, d_model=1024, nhead=8, num_layers=12, dim_feedforward=2048, scaler=None
    ):
        super().__init__()
        self.d_model = d_model
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.pos_encoder = nn.Embedding(50, d_model)  # Max seq len 50
        self.output_layer = nn.Linear(d_model, d_model)
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
        src = self.normalize(src)
        positions = (
            torch.arange(0, src.size(1), device=src.device)
            .unsqueeze(0)
            .repeat(src.size(0), 1)
        )
        src = src + self.pos_encoder(positions)
        mask = nn.Transformer.generate_square_subsequent_mask(src.size(1)).to(
            src.device
        )
        output = self.transformer(src, mask=mask)
        output = self.output_layer(output)
        return self.denormalize(output)


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
    nlp = spacy.load(f"{lang}_core_web_sm")
    doc = nlp(text)
    return [sent.text for sent in doc.sents]


def prepare_data(
    dataset_name,
    config_name=None,
    split="train",
    text_column="text",
    lang="eng_Latn",
    output_file="large_concept_model/data/processed/embeddings.npy",
    device="cpu",
    batch_size=1000,
):
    """
    Load a dataset from Hugging Face and prepare embeddings for LCM.

    Args:
        dataset_name (str): Name of the dataset on Hugging Face (e.g., 'wikitext').
        split (str): Dataset split to use (e.g., 'train', 'test').
        text_column (str): Name of the column containing text data.
        lang (str): Language code for SONAR (e.g., 'eng_Latn').
        output_file (str): Path to save the embeddings.
        device (str): Device to run SONAR encoding on ('cpu' or 'cuda').
        batch_size (int): Number of texts to process per batch.
    """
    # Load dataset from Hugging Face
    try:
        dataset = load_dataset(dataset_name, config_name, split=split)
        texts = dataset[text_column]
    except Exception as e:
        raise ValueError(
            f"Failed to load dataset '{dataset_name}' or access column '{text_column}': {str(e)}"
        )

    encoder = SonarEncoder(device=device)
    all_embeddings = []

    # Process texts in batches for efficiency
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        # Segment sentences using SpaCy
        sentences = [
            sent
            for text in batch_texts
            for sent in segment_sentences(text, lang.split("_")[0])
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
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        total_loss = 0
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)
            optimizer.zero_grad()
            pred = model(src)
            loss = criterion(pred, tgt)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader)}")


# Inference
def generate(model, initial_sequence, eot_embedding, max_length=50, threshold=0.9):
    sequence = initial_sequence.clone()
    for _ in range(max_length):
        pred = model(sequence)
        next_emb = pred[:, -1, :]
        sim_eot = torch.cosine_similarity(next_emb, eot_embedding, dim=-1)
        if sim_eot > threshold:
            break
        if sequence.size(1) > 1:
            sim_prev = torch.cosine_similarity(next_emb, sequence[:, -1, :], dim=-1)
            if sim_prev > threshold:
                break
        sequence = torch.cat([sequence, next_emb.unsqueeze(1)], dim=1)
    return sequence


# Main Execution
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Example: Load 'wikitext' dataset from Hugging Face
    dataset_name = "wikitext"
    config_name = "wikitext-2-raw-v1"  # Choose one of the available configs
    split = "train"
    text_column = "text"
    prepare_data(
        dataset_name,
        config_name=config_name,
        split=split,
        text_column=text_column,
        device=device,
    )

    # Load embeddings
    embeddings = torch.tensor(
        np.load("large_concept_model/data/processed/embeddings.npy"),
        dtype=torch.float32,
    ).to(device)
    eot_embedding = embeddings[-1].unsqueeze(0)  # "End of text." embedding

    # Fit scaler
    scaler = RobustScaler()
    scaler.fit(embeddings.cpu().numpy())

    # Initialize and train model
    model = LCMModel(scaler=scaler).to(device)
    dataset = EmbeddingDataset(embeddings)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    train(model, dataloader, device=device)

    # Perform inference
    decoder = SonarDecoder(device=device)
    initial_sequence = embeddings[:5].unsqueeze(0)  # Use first 5 embeddings as prompt
    generated_sequence = generate(model, initial_sequence, eot_embedding)
    generated_texts = decoder.decode(generated_sequence[0], "eng_Latn")
    print("Generated Text:", generated_texts)

    # Basic evaluation
    print("Evaluation: Check generated text coherence manually.")
