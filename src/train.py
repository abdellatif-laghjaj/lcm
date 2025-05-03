import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
import argparse
import spacy
import os
from tqdm.auto import tqdm
from datasets import load_dataset, Dataset as HFDataset  # Added HFDataset alias

# Assuming BaseLCM and SonarEncoder are in baselcm.py, ConceptSequenceDataset and add_noise_to_embeddings in utils.py
from baselcm import BaseLCM, SonarEncoder
from utils import ConceptSequenceDataset, add_noise_to_embeddings

# Set random seed for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


def parse_args():
    parser = argparse.ArgumentParser(description="Train the BaseLCM model")
    # Data Args
    parser.add_argument(
        "--hf_data",
        type=str,
        default="beomi/fineweb-edu-fortified-mini",
        help="Path or name of the Hugging Face dataset",
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="train",
        help="Dataset split to use (e.g., 'train', 'validation')",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default="text",
        help="Name of the text column in the dataset",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="en",
        help="Language code for Sonar encoding (e.g., 'en', 'fr')",
    )
    parser.add_argument(
        "--data_sample",
        type=int,
        default=1000,
        help="Number of samples to select from the dataset (for quick testing)",
    )
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=10,
        help="Length of input sequences for the model",
    )

    # Model Args
    parser.add_argument(
        "--input_dim",
        type=int,
        default=1024,
        help="Input dimension (embedding size from Sonar)",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=1024,
        help="Hidden dimension within the Transformer",
    )
    parser.add_argument(
        "--output_dim",
        type=int,
        default=1024,
        help="Output dimension (should match input_dim)",
    )
    parser.add_argument(
        "--num_heads", type=int, default=8, help="Number of attention heads"
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=12,
        help="Number of Transformer Decoder layers",
    )
    parser.add_argument(
        "--ff_dim",
        type=int,
        default=4096,
        help="Feedforward dimension in Transformer layers (often 4*hidden_dim)",
    )
    parser.add_argument(
        "--max_seq_len_model",
        type=int,
        default=50,
        help="Maximum sequence length the model's positional encoding can handle",
    )

    # Training Args
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for training"
    )
    parser.add_argument(
        "--encoding_batch_size",
        type=int,
        default=64,
        help="Batch size for sentence encoding (can be larger)",
    )
    parser.add_argument(
        "--epoch", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-5,
        help="Weight decay (L2 regularization)",
    )
    parser.add_argument(
        "--noise_level",
        type=float,
        default=0.05,
        help="Noise level added to input embeddings during training",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="saved_models",
        help="Directory to save trained models",
    )

    # Logging & Misc
    parser.add_argument(
        "--wandb", action="store_true", help="Enable Weights and Biases logging"
    )  # Use action='store_true'
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Force device ('cuda', 'cpu'). Auto-detects if None.",
    )

    return parser.parse_args()


# Centralized device management
def get_device(args_device: str | None) -> torch.device:
    if args_device:
        return torch.device(args_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Helper to load spacy model safely
def load_spacy_model(model_name="en_core_web_sm"):
    try:
        nlp = spacy.load(model_name)
    except OSError:
        print(f"Spacy model '{model_name}' not found. Downloading...")
        spacy.cli.download(model_name)
        nlp = spacy.load(model_name)
    return nlp


def train(args):
    device = get_device(args.device)
    print(f"Using device: {device}")

    if args.wandb:
        wandb.init(project="base-lcm", config=args)

    # --- 1. Load and Prepare Data ---
    print(f"Loading dataset: {args.hf_data} [{args.dataset_split}]")
    # Use streaming=True for very large datasets if memory becomes an issue during loading
    dataset = load_dataset(args.hf_data, split=args.dataset_split)

    # Select a subset if specified
    if args.data_sample is not None and args.data_sample < len(dataset):
        print(f"Selecting {args.data_sample} samples from the dataset.")
        dataset = dataset.select(range(args.data_sample))

    # Load spacy model for sentence splitting
    print("Loading spacy model for sentence splitting...")
    nlp = load_spacy_model()

    # Function to split text into sentences
    def split_into_sentences(example):
        # example[args.text_column] is a list of texts when batched=True
        texts_batch = example[args.text_column]
        # Use nlp.pipe for efficient processing of the batch
        docs = nlp.pipe(texts_batch)
        # Create a list of lists: outer list corresponds to batch items, inner lists contain sentences for each item
        batch_sentences = [
            [sent.text.strip() for sent in doc.sents if sent.text.strip()]
            for doc in docs
        ]
        return {"sentences": batch_sentences}

    # Split texts into sentences - Use map for efficiency
    print("Splitting corpus into sentences...")
    sentence_dataset = dataset.map(
        split_into_sentences,
        batched=True,
        batch_size=100,
        remove_columns=dataset.column_names,
    )

    # Flatten the list of lists of sentences
    all_sentences = [
        sentence for sublist in sentence_dataset["sentences"] for sentence in sublist
    ]
    print(f"Total number of sentences: {len(all_sentences)}")

    if not all_sentences:
        print("No sentences found after processing. Check dataset and text_column.")
        return

    # --- 2. Encode Sentences ---
    print("Initializing SonarEncoder...")
    # Pass the determined device to SonarEncoder
    encoder = SonarEncoder(device=str(device))  # Ensure device is string

    print("Encoding sentences...")
    # Ensure embeddings end up on the correct device *after* potential CPU offloading in encode()
    all_embeddings = encoder.encode(
        all_sentences, lang=args.lang, batch_size=args.encoding_batch_size
    ).to(device)

    # Clear memory
    del encoder, dataset, sentence_dataset, all_sentences
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"Embeddings shape: {all_embeddings.shape}")

    if len(all_embeddings) <= args.sequence_length:
        print(
            f"Error: Number of embeddings ({len(all_embeddings)}) is less than or equal to sequence length ({args.sequence_length}). Cannot create sequences."
        )
        return

    # --- 3. Create Dataset and DataLoader ---
    print("Creating sequence dataset and dataloader...")
    train_dataset = ConceptSequenceDataset(all_embeddings, args.sequence_length)
    # Use persistent_workers and pin_memory for potential speedup if GPU is available
    num_workers = 4 if device.type == "cuda" else 0
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        # persistent_workers=(num_workers > 0) # Can uncomment if needed
    )
    print(f"Dataset size: {len(train_dataset)} sequences")

    # --- 4. Initialize Model, Optimizer, Criterion ---
    print("Initializing BaseLCM model...")
    model = BaseLCM(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        ff_dim=args.ff_dim,
        output_dim=args.output_dim,
        max_seq_len=args.max_seq_len_model,  # Pass max sequence length
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    criterion = nn.MSELoss()

    # --- 5. Training Loop ---
    print("Starting training...")
    for epoch in range(args.epoch):
        model.train()
        running_loss = 0.0

        # Wrap dataloader with tqdm for progress bar
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.epoch}", leave=False)
        for batch_idx, (input_seq, target_emb) in enumerate(pbar):
            # Move data to device
            input_seq = input_seq.to(device)
            target_emb = target_emb.to(device)

            # Add noise to input sequence
            noisy_input_seq = add_noise_to_embeddings(input_seq, args.noise_level)

            optimizer.zero_grad()

            # Get model prediction (predicts embedding for the *next* step after the sequence)
            # The model outputs predictions for each position in the input sequence.
            # We are interested in the prediction based on the *entire* input sequence,
            # which corresponds to the output at the *last* position of the sequence.
            output_embeddings_seq = model(noisy_input_seq)
            predicted_next_emb = output_embeddings_seq[
                :, -1, :
            ]  # Get prediction from the last time step

            loss = criterion(predicted_next_emb, target_emb)

            loss.backward()
            # Optional: Gradient clipping
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()

            # Update progress bar postfix
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                avg_loss=f"{running_loss / (batch_idx + 1):.4f}",
            )

        epoch_loss = running_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{args.epoch} - Average Loss: {epoch_loss:.4f}")

        if args.wandb:
            wandb.log({"epoch": epoch + 1, "loss": epoch_loss})

    print("Training Complete!")

    # --- 6. Save Model ---
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, "base_lcm_model.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved at {save_path}")

    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    train(args)
